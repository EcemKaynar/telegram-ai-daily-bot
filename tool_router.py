import json

from openrouter_client import (
    get_tool_call_decision,
    get_json_tool_decision
)


def parse_tool_arguments(arguments):
    if not arguments:
        return {}

    if isinstance(arguments, dict):
        return arguments

    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except Exception:
            return {}

    return {}


def extract_city_from_tool_calls(tool_calls):
    if not tool_calls:
        return None, None

    first_tool_call = tool_calls[0]

    function_data = first_tool_call.get("function", {})
    function_name = function_data.get("name", "")
    arguments = parse_tool_arguments(function_data.get("arguments"))

    if function_name == "get_weather":
        city = arguments.get("city", "")

        if city:
            return city.strip(), first_tool_call

    return None, first_tool_call


def detect_tool_call(user_text, history_context=""):
    """
    Bu fonksiyon artık şehir listesine bağlı çalışmaz.
    LLM'den tool decision ister.
    Beklenen çıktı:
    - weather sorusuysa tool='weather', city='Paris/London/New York/...'
    - değilse tool=None
    """

    try:
        decision = get_tool_call_decision(
            user_text=user_text,
            history_context=history_context
        )

        tool_calls = decision.get("tool_calls", [])
        city, raw_tool_call = extract_city_from_tool_calls(tool_calls)

        if city:
            return {
                "tool": "weather",
                "city": city,
                "assistant_message": decision.get("assistant_message", ""),
                "raw_tool_call": raw_tool_call,
                "raw_decision": decision
            }

    except Exception as error:
        print(f"Tool call decision hatası: {error}")

    try:
        json_decision = get_json_tool_decision(
            user_text=user_text,
            history_context=history_context
        )

        if json_decision.get("tool") == "weather":
            city = json_decision.get("city", "").strip()

            return {
                "tool": "weather",
                "city": city,
                "assistant_message": "",
                "raw_tool_call": None,
                "raw_decision": json_decision
            }

    except Exception as error:
        print(f"JSON tool decision hatası: {error}")

    return {
        "tool": None,
        "city": None,
        "assistant_message": "",
        "raw_tool_call": None,
        "raw_decision": None
    }