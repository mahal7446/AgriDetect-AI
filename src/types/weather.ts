export interface WeatherData {
    temperature: number;
    humidity: number;
    condition: string;
    windSpeed: number;
    location: string;
    recommendation: string;
    feelsLike: number;
}

export interface OpenWeatherResponse {
    main: {
        temp: number;
        feels_like: number;
        humidity: number;
    };
    weather: Array<{
        main: string;
        description: string;
    }>;
    wind: {
        speed: number;
    };
    name: string;
}
