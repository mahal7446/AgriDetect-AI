import { useState, useEffect } from 'react';
import { useGeolocation } from './useGeolocation';
import type { WeatherData, OpenWeatherResponse } from '@/types/weather';

const API_KEY = import.meta.env.VITE_OPENWEATHER_API_KEY;
const CACHE_DURATION = 10 * 60 * 1000; // 10 minutes in milliseconds

interface WeatherState {
    data: WeatherData | null;
    loading: boolean;
    error: string | null;
}

// Generate agriculture-focused recommendations based on weather
const getRecommendation = (temp: number, humidity: number, windSpeed: number): string => {
    if (humidity > 80) {
        return 'High humidity – monitor for fungal diseases like leaf blight and powdery mildew';
    } else if (humidity < 30) {
        return 'Low humidity – ensure adequate irrigation for optimal crop growth';
    } else if (temp > 35) {
        return 'High temperature – watch for heat stress and ensure proper watering';
    } else if (temp < 10) {
        return 'Low temperature – protect crops from potential frost damage';
    } else if (windSpeed > 20) {
        return 'Strong winds – check for physical damage to plants and secure supports';
    } else if (humidity >= 60 && humidity <= 70 && temp >= 20 && temp <= 30) {
        return 'Ideal conditions for plant growth – great time for field activities';
    }
    return 'Good farming conditions – monitor crops regularly for any issues';
};

interface UseWeatherOptions {
    enabled?: boolean;
}

export const useWeather = (options: UseWeatherOptions = {}) => {
    const { latitude, longitude, error: geoError, loading: geoLoading } = useGeolocation({ enabled: options.enabled });
    const [state, setState] = useState<WeatherState>({
        data: null,
        loading: true,
        error: null,
    });

    useEffect(() => {
        // If geolocation is still loading or has an error, update state accordingly
        if (geoLoading) {
            setState(prev => ({ ...prev, loading: true }));
            return;
        }

        if (geoError) {
            setState({
                data: null,
                loading: false,
                error: geoError,
            });
            return;
        }

        if (!latitude || !longitude) {
            return;
        }

        // Check cache first using coordinates-specific key
        const cacheKey = `weather_cache_${latitude}_${longitude}`;
        const cacheTimeKey = `weather_cache_time_${latitude}_${longitude}`;
        const cachedData = localStorage.getItem(cacheKey);
        const cachedTime = localStorage.getItem(cacheTimeKey);

        if (cachedData && cachedTime) {
            const timeDiff = Date.now() - parseInt(cachedTime);
            if (timeDiff < CACHE_DURATION) {
                const parsed = JSON.parse(cachedData) as WeatherData;
                setState({
                    data: parsed,
                    loading: false,
                    error: null,
                });
                return;
            }
        }

        // Fetch fresh weather data
        const fetchWeather = async () => {
            console.log('🌤️ Fetching weather...', { latitude, longitude, API_KEY: API_KEY ? 'Present' : 'Missing' });

            if (!API_KEY) {
                console.error('❌ API key is missing');
                setState({
                    data: null,
                    loading: false,
                    error: 'Weather API key is not configured',
                });
                return;
            }

            try {
                const url = `https://api.openweathermap.org/data/2.5/weather?lat=${latitude}&lon=${longitude}&units=metric&appid=${API_KEY}`;
                console.log('🌐 API URL:', url.replace(API_KEY, 'API_KEY_HIDDEN'));

                const response = await fetch(url);
                console.log('📡 Response status:', response.status, response.statusText);

                if (!response.ok) {
                    const errorText = await response.text();
                    console.error('❌ API Error:', errorText);
                    throw new Error(`API returned ${response.status}: ${errorText}`);
                }

                const data: OpenWeatherResponse = await response.json();
                console.log('✅ Weather data received:', data);

                const weatherData: WeatherData = {
                    temperature: Math.round(data.main.temp),
                    feelsLike: Math.round(data.main.feels_like),
                    humidity: data.main.humidity,
                    condition: data.weather[0].main,
                    windSpeed: Math.round(data.wind.speed * 3.6), // Convert m/s to km/h
                    location: data.name,
                    recommendation: getRecommendation(
                        data.main.temp,
                        data.main.humidity,
                        data.wind.speed * 3.6
                    ),
                };

                // Cache the data per coordinates
                localStorage.setItem(cacheKey, JSON.stringify(weatherData));
                localStorage.setItem(cacheTimeKey, Date.now().toString());

                setState({
                    data: weatherData,
                    loading: false,
                    error: null,
                });
            } catch (err) {
                console.error('❌ Weather fetch error:', err);
                setState({
                    data: null,
                    loading: false,
                    error: err instanceof Error ? err.message : 'Failed to load weather data',
                });
            }
        };

        fetchWeather();
    }, [latitude, longitude, geoError, geoLoading]);

    return state;
};
