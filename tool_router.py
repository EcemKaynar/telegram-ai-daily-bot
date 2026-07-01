import json

from openrouter_client import get_tool_call_decision


def parse_tool_arguments(arguments):
    if not arguments:
        return {}

    if isinstance(arguments, dict):
        return arguments

    try:
        return json.loads(arguments)
    except Exception:
        return {}


def detect_tool_call(user_text, history_context=""):
    """
    Manuel keyword veya şehir listesi kullanmaz.
    Kararı OpenRouter modelinden alır.
    """

    tool_router_input = user_text

    if history_context:
        tool_router_input = (
            f"Previous conversation context:\n{history_context}\n\n"
            f"Current user message:\n{user_text}\n\n"
            "Use previous context only to understand missing information like city or user preference."
        )

    message = get_tool_call_decision(tool_router_input)

    tool_calls = message.get("tool_calls") or []

    if tool_calls:
        first_tool_call = tool_calls[0]
        function_data = first_tool_call.get("function", {})

        tool_name = function_data.get("name")
        arguments = parse_tool_arguments(function_data.get("arguments"))

        if tool_name == "get_current_weather":
            city = arguments.get("city") or arguments.get("location")

            return {
                "tool": "weather",
                "city": city,
                "assistant_message": message,
                "raw_tool_call": first_tool_call
            }

    json_decision = message.get("json_decision")

    if json_decision:
        tool = json_decision.get("tool")
        city = json_decision.get("city")

        if tool == "weather" and city:
            return {
                "tool": "weather",
                "city": city,
                "assistant_message": None,
                "raw_tool_call": None
            }

    return {
        "tool": "none",
        "city": None,
        "assistant_message": None,
        "raw_tool_call": None
    }