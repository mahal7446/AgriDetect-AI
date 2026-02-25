import { useState, useEffect } from 'react';

interface GeolocationState {
    latitude: number | null;
    longitude: number | null;
    error: string | null;
    loading: boolean;
}

interface UseGeolocationOptions {
    enabled?: boolean;
}

export const useGeolocation = (options: UseGeolocationOptions = {}) => {
    const { enabled = true } = options;
    const [state, setState] = useState<GeolocationState>({
        latitude: null,
        longitude: null,
        error: null,
        loading: true,
    });

    useEffect(() => {
        let watchId: number | null = null;
        // If location access is disabled, use default location immediately
        if (!enabled) {
            console.log('📍 Real-time location disabled, using default location: New Delhi');
            setState({
                latitude: 28.6139,
                longitude: 77.2090,
                error: null,
                loading: false,
            });
            return;
        }
        if (!navigator.geolocation) {
            console.error('❌ Geolocation not supported');
            setState({
                latitude: null,
                longitude: null,
                error: 'Your browser does not support geolocation. Please use a modern browser to access location-based weather.',
                loading: false,
            });
            return;
        }

        console.log('📍 Requesting geolocation...');

        const onSuccess = (position: GeolocationPosition) => {
            console.log('✅ Geolocation success:', {
                latitude: position.coords.latitude,
                longitude: position.coords.longitude
            });
            setState({
                latitude: position.coords.latitude,
                longitude: position.coords.longitude,
                error: null,
                loading: false,
            });
        };

        const onError = (error: GeolocationPositionError) => {
            let errorMessage = 'Unable to retrieve your location.';

            switch (error.code) {
                case error.PERMISSION_DENIED:
                    errorMessage = 'Location permission was denied. Please enable location access in your browser settings to see weather for your area.';
                    console.warn('⚠️ Location permission denied by user');
                    break;
                case error.POSITION_UNAVAILABLE:
                    errorMessage = 'Your location information is currently unavailable. Please check your device\'s location services are turned on.';
                    break;
                case error.TIMEOUT:
                    errorMessage = 'The request to get your location timed out. Please check your internet connection and try again.';
                    break;
            }

            console.error('❌ Geolocation error:', errorMessage, error);

            setState({
                latitude: null,
                longitude: null,
                error: errorMessage,
                loading: false,
            });
        };

        navigator.geolocation.getCurrentPosition(onSuccess, onError, {
            enableHighAccuracy: true,
            timeout: 10000,
            maximumAge: 300000, // Cache position for 5 minutes
        });

        // Watch for changes to update weather when user moves
        watchId = navigator.geolocation.watchPosition(onSuccess, onError, {
            enableHighAccuracy: true,
            maximumAge: 300000,
        });

        return () => {
            if (watchId !== null) {
                navigator.geolocation.clearWatch(watchId);
            }
        };
    }, [enabled]);

    return state;
};
