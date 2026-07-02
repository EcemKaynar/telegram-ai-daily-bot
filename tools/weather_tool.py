import json
import os
import time

import requests


WEATHER_PROVIDER = os.getenv("WEATHER_PROVIDER", "auto").lower()


def to_json_text(data):
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


def now_ms():
    return time.perf_counter()


def elapsed_ms(start):
    return round((time.perf_counter() - start) * 1000, 2)


def get_coordinates_by_city(city_name):
    start = now_ms()

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city_name,
        "count": 1,
        "language": "tr",
        "format": "json"
    }

    response = requests.get(url, params=params, timeout=10)
    duration = elapsed_ms(start)

    response.raise_for_status()

    data = response.json()

    log = {
        "request": {
            "provider": "open_meteo_geocoding",
            "url": url,
            "params": params
        },
        "response": data,
        "duration_ms": duration
    }

    results = data.get("results", [])

    if not results:
        return None, log

    first = results[0]

    return {
        "name": first.get("name"),
        "country": first.get("country"),
        "latitude": first.get("latitude"),
        "longitude": first.get("longitude"),
        "timezone": first.get("timezone")
    }, log


def get_weather_open_meteo(city_name):
    logs = []

    coordinates, geo_log = get_coordinates_by_city(city_name)
    logs.append(geo_log)

    if not coordinates:
        return {
            "success": False,
            "provider": "open_meteo",
            "message": f"{city_name} için konum bulunamadı.",
            "data": None,
            "logs": logs
        }

    start = now_ms()

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": coordinates["latitude"],
        "longitude": coordinates["longitude"],
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m",
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=10)
    duration = elapsed_ms(start)

    response.raise_for_status()

    data = response.json()

    logs.append({
        "request": {
            "provider": "open_meteo_forecast",
            "url": url,
            "params": params
        },
        "response": data,
        "duration_ms": duration
    })

    current = data.get("current", {})

    weather_data = {
        "provider": "open_meteo",
        "city": coordinates.get("name"),
        "country": coordinates.get("country"),
        "latitude": coordinates.get("latitude"),
        "longitude": coordinates.get("longitude"),
        "temperature_c": current.get("temperature_2m"),
        "humidity_percent": current.get("relative_humidity_2m"),
        "precipitation_mm": current.get("precipitation"),
        "rain_mm": current.get("rain"),
        "weather_code": current.get("weather_code"),
        "wind_speed_kmh": current.get("wind_speed_10m"),
        "time": current.get("time")
    }

    return {
        "success": True,
        "provider": "open_meteo",
        "message": "Hava durumu başarıyla alındı.",
        "data": weather_data,
        "logs": logs
    }


def get_weather_wttr(city_name):
    logs = []
    start = now_ms()

    url = f"https://wttr.in/{city_name}"

    params = {
        "format": "j1"
    }

    response = requests.get(url, params=params, timeout=10)
    duration = elapsed_ms(start)

    response.raise_for_status()

    data = response.json()

    logs.append({
        "request": {
            "provider": "wttr_in",
            "url": url,
            "params": params
        },
        "response": data,
        "duration_ms": duration
    })

    current_condition = data.get("current_condition", [{}])[0]
    nearest_area = data.get("nearest_area", [{}])[0]

    area_name = ""
    country = ""

    if nearest_area:
        area_name_items = nearest_area.get("areaName", [])
        country_items = nearest_area.get("country", [])

        if area_name_items:
            area_name = area_name_items[0].get("value", "")

        if country_items:
            country = country_items[0].get("value", "")

    weather_data = {
        "provider": "wttr_in",
        "city": area_name or city_name,
        "country": country,
        "temperature_c": current_condition.get("temp_C"),
        "feels_like_c": current_condition.get("FeelsLikeC"),
        "humidity_percent": current_condition.get("humidity"),
        "precipitation_mm": current_condition.get("precipMM"),
        "weather_description": current_condition.get("weatherDesc", [{}])[0].get("value", ""),
        "wind_speed_kmh": current_condition.get("windspeedKmph"),
        "observation_time": current_condition.get("observation_time")
    }

    return {
        "success": True,
        "provider": "wttr_in",
        "message": "Hava durumu başarıyla alındı.",
        "data": weather_data,
        "logs": logs
    }


def get_weather_by_city(city_name, provider=None):
    selected_provider = (provider or WEATHER_PROVIDER or "auto").lower()

    if selected_provider == "open_meteo":
        try:
            return get_weather_open_meteo(city_name)
        except Exception as error:
            return {
                "success": False,
                "provider": "open_meteo",
                "message": f"Open-Meteo hava durumu alınamadı: {error}",
                "data": None,
                "logs": []
            }

    if selected_provider == "wttr":
        try:
            return get_weather_wttr(city_name)
        except Exception as error:
            return {
                "success": False,
                "provider": "wttr_in",
                "message": f"wttr.in hava durumu alınamadı: {error}",
                "data": None,
                "logs": []
            }

    all_logs = []

    try:
        result = get_weather_open_meteo(city_name)

        all_logs.extend(result.get("logs", []))

        if result["success"]:
            result["logs"] = all_logs
            return result

    except Exception as error:
        all_logs.append({
            "request": {
                "provider": "open_meteo",
                "city": city_name
            },
            "response": {
                "error": str(error)
            },
            "duration_ms": None
        })

    try:
        result = get_weather_wttr(city_name)

        all_logs.extend(result.get("logs", []))

        if result["success"]:
            result["logs"] = all_logs
            return result

    except Exception as error:
        all_logs.append({
            "request": {
                "provider": "wttr_in",
                "city": city_name
            },
            "response": {
                "error": str(error)
            },
            "duration_ms": None
        })

    return {
        "success": False,
        "provider": "auto",
        "message": "Hava durumu servislerinden cevap alınamadı.",
        "data": None,
        "logs": all_logs
    }