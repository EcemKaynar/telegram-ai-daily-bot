import json
import requests


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"


WEATHER_CODE_MAP = {
    0: "Açık",
    1: "Çoğunlukla açık",
    2: "Parçalı bulutlu",
    3: "Kapalı",
    45: "Sisli",
    48: "Kırağılı sis",
    51: "Hafif çisenti",
    53: "Orta çisenti",
    55: "Yoğun çisenti",
    61: "Hafif yağmur",
    63: "Orta yağmur",
    65: "Şiddetli yağmur",
    71: "Hafif kar",
    73: "Orta kar",
    75: "Yoğun kar",
    80: "Hafif sağanak",
    81: "Orta sağanak",
    82: "Şiddetli sağanak",
    95: "Gök gürültülü fırtına"
}


def get_coordinates(city_name):
    params = {
        "name": city_name,
        "count": 1,
        "language": "tr",
        "format": "json"
    }

    response = requests.get(GEOCODING_URL, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()

    if "results" not in data or len(data["results"]) == 0:
        return None, {
            "request": {
                "url": GEOCODING_URL,
                "params": params
            },
            "response": data
        }

    location = data["results"][0]

    result = {
        "name": location.get("name"),
        "country": location.get("country"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "timezone": location.get("timezone")
    }

    return result, {
        "request": {
            "url": GEOCODING_URL,
            "params": params
        },
        "response": data
    }


def get_weather_by_city(city_name):
    try:
        location, geocoding_log = get_coordinates(city_name)

        if not location:
            return {
                "success": False,
                "message": f"{city_name} için konum bulunamadı.",
                "logs": [geocoding_log]
            }

        params = {
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "current": "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
            "timezone": "auto",
            "forecast_days": 1
        }

        response = requests.get(WEATHER_URL, params=params, timeout=30)
        response.raise_for_status()

        weather_data = response.json()
        current = weather_data.get("current", {})
        daily = weather_data.get("daily", {})

        weather_code = current.get("weather_code")

        formatted_result = {
            "success": True,
            "city": location["name"],
            "country": location["country"],
            "temperature": current.get("temperature_2m"),
            "apparent_temperature": current.get("apparent_temperature"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m"),
            "weather_code": weather_code,
            "weather_description": WEATHER_CODE_MAP.get(weather_code, "Bilinmiyor"),
            "max_temperature": daily.get("temperature_2m_max", [None])[0],
            "min_temperature": daily.get("temperature_2m_min", [None])[0],
            "precipitation_probability": daily.get("precipitation_probability_max", [None])[0],
            "raw": weather_data
        }

        weather_log = {
            "request": {
                "url": WEATHER_URL,
                "params": params
            },
            "response": weather_data
        }

        return {
            "success": True,
            "data": formatted_result,
            "logs": [geocoding_log, weather_log]
        }

    except Exception as error:
        return {
            "success": False,
            "message": f"Hava durumu servisine bağlanırken hata oluştu: {error}",
            "logs": []
        }


def to_json_text(data):
    return json.dumps(data, ensure_ascii=False)