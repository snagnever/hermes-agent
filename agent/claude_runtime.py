"""claude_agent_sdk runtime path.

* run_claude_agent_sdk_turn — drives one turn through a ClaudeAgentSession
  (Claude Agent SDK / Claude Code subprocess) and projects the results back
  into Hermes' messages + usage accounting.

Sibling of agent/codex_runtime.py — keep the two in structural lockstep.
Called from run_conversation() when agent.api_mode == "claude_agent_sdk".
Returns the same dict shape as the codex_app_server path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def _coerce_usage_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value), 0)
        except ValueError:
            return 0
    return 0


def _record_claude_agent_usage(agent, turn) -> dict[str, Any]:
    """Translate Agent SDK ResultMessage.usage into Hermes accounting.

    SDK keys (Anthropic shape): input_tokens (uncached), output_tokens,
    cache_read_input_tokens, cache_creation_input_tokens. Hermes' canonical
    prompt bucket = uncached input + cache-read + cache-write. Mirrors
    _record_codex_app_server_usage.

    Even when the SDK omits usage for a turn, Hermes still counts that turn
    as one API call for session/status accounting.
    """
    agent.session_api_calls += 1

    usage = getattr(turn, "usage", None)
    if not isinstance(usage, dict) or not usage:
        if agent._session_db and agent.session_id:
            try:
                if not agent._session_db_created:
                    agent._ensure_db_session()
                agent._session_db.update_token_counts(
                    agent.session_id,
                    model=agent.model,
                    api_call_count=1,
                )
            except Exception as exc:
                logger.debug(
                    "claude-agent api-call persistence failed (session=%s): %s",
                    agent.session_id, exc,
                )
        return {}

    from agent.usage_pricing import CanonicalUsage, estimate_usage_cost

    input_tokens = _coerce_usage_int(usage.get("input_tokens"))
    cache_read_tokens = _coerce_usage_int(usage.get("cache_read_input_tokens"))
    cache_write_tokens = _coerce_usage_int(usage.get("cache_creation_input_tokens"))
    output_tokens = _coerce_usage_int(usage.get("output_tokens"))

    canonical_usage = CanonicalUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        reasoning_tokens=0,
        raw_usage=usage,
    )
    prompt_tokens = canonical_usage.prompt_tokens
    completion_tokens = canonical_usage.output_tokens
    total_tokens = canonical_usage.total_tokens
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": canonical_usage.input_tokens,
        "output_tokens": canonical_usage.output_tokens,
        "cache_read_tokens": canonical_usage.cache_read_tokens,
        "cache_write_tokens": canonical_usage.cache_write_tokens,
        "reasoning_tokens": canonical_usage.reasoning_tokens,
    }

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            compressor.update_from_response(usage_dict)
        except Exception:
            logger.debug("claude-agent usage update failed", exc_info=True)

    agent.session_prompt_tokens += prompt_tokens
    agent.session_completion_tokens += completion_tokens
    agent.session_total_tokens += total_tokens
    agent.session_input_tokens += canonical_usage.input_tokens
    agent.session_output_tokens += canonical_usage.output_tokens
    agent.session_cache_read_tokens += canonical_usage.cache_read_tokens
    agent.session_cache_write_tokens += canonical_usage.cache_write_tokens
    agent.session_reasoning_tokens += canonical_usage.reasoning_tokens

    cost_result = estimate_usage_cost(
        agent.model,
        canonical_usage,
        provider=agent.provider,
        base_url=agent.base_url,
        api_key=getattr(agent, "api_key", ""),
    )
    if cost_result.amount_usd is not None:
        agent.session_estimated_cost_usd += float(cost_result.amount_usd)
    agent.session_cost_status = cost_result.status
    agent.session_cost_source = cost_result.source

    if agent._session_db and agent.session_id:
        try:
            if not agent._session_db_created:
                agent._ensure_db_session()
            agent._session_db.update_token_counts(
                agent.session_id,
                input_tokens=canonical_usage.input_tokens,
                output_tokens=canonical_usage.output_tokens,
                cache_read_tokens=canonical_usage.cache_read_tokens,
                cache_write_tokens=canonical_usage.cache_write_tokens,
                reasoning_tokens=canonical_usage.reasoning_tokens,
                estimated_cost_usd=float(cost_result.amount_usd)
                if cost_result.amount_usd is not None else None,
                cost_status=cost_result.status,
                cost_source=cost_result.source,
                billing_provider=agent.provider,
                billing_base_url=agent.base_url,
                billing_mode="subscription_included"
                if cost_result.status == "included" else None,
                model=agent.model,
                api_call_count=1,
            )
        except Exception as exc:
            logger.debug(
                "claude-agent token persistence failed (session=%s, tokens=%d): %s",
                agent.session_id, total_tokens, exc,
            )

    return {
        **usage_dict,
        "last_prompt_tokens": prompt_tokens,
        "estimated_cost_usd": float(cost_result.amount_usd)
        if cost_result.amount_usd is not None else None,
        "cost_status": cost_result.status,
        "cost_source": cost_result.source,
    }


def run_claude_agent_sdk_turn(
    agent,
    *,
    user_message: str,
    original_user_message: Any,
    messages: List[Dict[str, Any]],
    effective_task_id: str,
    should_review_memory: bool = False,
) -> Dict[str, Any]:
    """Claude Agent SDK runtime path. Hands the entire turn to a Claude Code
    subprocess (via ClaudeAgentSession) and projects its events back into
    Hermes' messages list so memory/skill review keep working."""
    from agent.transports.claude_agent_session import ClaudeAgentSession

    # Lazy session: one ClaudeAgentSession per AIAgent instance. Spawned on
    # first turn, reused across turns, closed at AIAgent shutdown.
    if getattr(agent, "_claude_session", None) is None:
        from agent.runtime_cwd import resolve_agent_cwd

        cwd = getattr(agent, "session_cwd", None) or str(resolve_agent_cwd())
        # Approval callback: defer to Hermes' standard prompt flow if a CLI
        # thread has installed one. Gateway / cron contexts get the
        # fail-closed default.
        try:
            from tools.terminal_tool import _get_approval_callback

            approval_callback = _get_approval_callback()
        except Exception:
            approval_callback = None

        # When the user has explicitly opted out of Hermes approvals (yolo /
        # approvals.mode: off), let Claude Code's bypassPermissions mode be the
        # policy gate. Defaults (manual/smart/unset) preserve fail-closed.
        auto_approve = False
        try:
            from tools.approval import is_approval_bypass_active

            auto_approve = is_approval_bypass_active()
        except Exception:
            logger.debug(
                "claude-agent: approval-bypass lookup failed; fail-closed",
                exc_info=True,
            )

        session = ClaudeAgentSession(
            model=agent.model,
            cwd=cwd,
            system_prompt=getattr(agent, "system_prompt", None),
            approval_callback=approval_callback,
            auto_approve=auto_approve,
        )
        # Adapt the session's 3-arg (name, preview, args) progress hook onto
        # Hermes' 4-arg (event_type, tool_name, preview, args) callback so
        # gateways show verbose "running X" breadcrumbs on this route too.
        agent_progress = getattr(agent, "tool_progress_callback", None)
        if agent_progress is not None:
            def _emit(name, preview, args, _cb=agent_progress):
                try:
                    _cb("tool.started", name, preview, args)
                except Exception:
                    logger.debug("claude tool-progress callback raised", exc_info=True)

            session.tool_progress_callback = _emit
        agent._claude_session = session

    session = agent._claude_session

    # NOTE: the inbound user message is ALREADY appended to messages by the
    # standard run_conversation() flow before this early-return path. Do NOT
    # append it again.
    turn = session.run_turn(user_message)

    # run_turn never raises (errors land in turn.error), but if it signalled
    # the session is wedged, retire it so the next turn respawns Claude Code.
    if getattr(turn, "should_retire", False):
        logger.warning("claude-agent session retired (turn error: %s)", turn.error)
        try:
            session.close()
        except Exception:
            pass
        agent._claude_session = None

    # Splice projected messages into the conversation. run_turn already emits
    # the assistant text row(s) plus tool_call/tool pairs, so final_text is
    # NOT appended separately (that would duplicate the last assistant turn).
    if turn.projected_messages:
        messages.extend(turn.projected_messages)
        # This early-return path bypasses conversation_loop's per-step
        # persistence, so flush the projected rows ourselves. The inbound user
        # turn was already flushed at turn start; the flush dedups via the
        # intrinsic _DB_PERSISTED_MARKER, so we write only the new rows and
        # can report agent_persisted=True (gateway then skips its own write).
        if getattr(agent, "_session_db", None) is not None:
            try:
                agent._flush_messages_to_session_db(messages)
            except Exception:
                logger.debug(
                    "claude-agent projected-message flush failed", exc_info=True
                )

    # Counter ticks for the agent-improvement loop. _turns_since_memory /
    # _user_turn_count are already bumped in run_conversation's pre-loop
    # block; only _iters_since_skill needs an explicit bump here since the
    # chat_completions tool loop (which normally bumps it) is bypassed.
    agent._iters_since_skill = (
        getattr(agent, "_iters_since_skill", 0) + turn.tool_iterations
    )
    usage_result = _record_claude_agent_usage(agent, turn)

    should_review_skills = False
    if (
        agent._skill_nudge_interval > 0
        and agent._iters_since_skill >= agent._skill_nudge_interval
        and "skill_manage" in agent.valid_tool_names
    ):
        should_review_skills = True
        agent._iters_since_skill = 0

    # External memory sync — skipped on interrupt/error to avoid feeding
    # partial transcripts to memory.
    if not turn.interrupted and turn.error is None:
        try:
            agent._sync_external_memory_for_turn(
                original_user_message=original_user_message,
                final_response=turn.final_text,
                interrupted=False,
                messages=messages,
            )
        except Exception:
            logger.debug("external memory sync raised", exc_info=True)

    # Background review fork — same cadence + signature as the default path.
    if (
        turn.final_text
        and not turn.interrupted
        and (should_review_memory or should_review_skills)
    ):
        try:
            agent._spawn_background_review(
                messages_snapshot=list(messages),
                review_memory=should_review_memory,
                review_skills=should_review_skills,
            )
        except Exception:
            logger.debug("background review spawn raised", exc_info=True)

    return {
        "final_response": turn.final_text,
        "messages": messages,
        "api_calls": 1,
        "completed": not turn.interrupted and turn.error is None,
        "partial": turn.interrupted or turn.error is not None,
        "error": turn.error,
        "agent_persisted": True,
        "claude_session_id": turn.session_id,
        **usage_result,
    }


__all__ = ["run_claude_agent_sdk_turn"]
