import { useState } from "react";
import { Cloud, Droplets, Wind, Sun, ThermometerSun, AlertTriangle, MapPin, Loader2, RefreshCw } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { useWeather } from "@/hooks/useWeather";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

export const WeatherWidget = () => {
  const [useRealTimeLocation, setUseRealTimeLocation] = useState(true);
  const { data: weather, loading, error } = useWeather({ enabled: useRealTimeLocation });

  // Loading state
  if (loading) {
    return (
      <Card className="overflow-hidden border-border/50 shadow-soft">
        <div className="gradient-sky p-1">
          <CardContent className="p-4 bg-card/95 backdrop-blur-sm rounded-lg">
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-8 h-8 text-primary animate-spin" />
              <p className="ml-3 text-muted-foreground">Loading weather data...</p>
            </div>
          </CardContent>
        </div>
      </Card>
    );
  }

  // Error state
  if (error || !weather) {
    const isLocationError = error?.toLowerCase().includes('location') || error?.toLowerCase().includes('permission');
    return (
      <Card className="overflow-hidden border-border/50 shadow-soft">
        <div className="gradient-sky p-1">
          <CardContent className="p-4 bg-card/95 backdrop-blur-sm rounded-lg">
            {/* Location Toggle - show even in error state */}
            <div className="flex items-center justify-between mb-4 pb-3 border-b border-border/50">
              <div className="flex items-center gap-2">
                <MapPin className="w-4 h-4 text-muted-foreground" />
                <Label htmlFor="location-toggle-error" className="text-sm font-medium cursor-pointer">
                  Current Location Weather
                </Label>
              </div>
              <div className="flex items-center gap-3 p-2 rounded-full bg-muted/30">
                <span
                  className={`text-xs font-bold transition-all duration-300 ${useRealTimeLocation
                    ? 'text-muted-foreground/50 scale-90'
                    : 'text-red-500 scale-110 drop-shadow-[0_0_8px_rgba(239,68,68,0.6)]'
                    }`}
                >
                  OFF
                </span>
                <div className={`relative transition-all duration-300 ${useRealTimeLocation
                  ? 'drop-shadow-[0_0_12px_rgba(34,197,94,0.8)]'
                  : ''
                  }`}>
                  <Switch
                    id="location-toggle-error"
                    checked={useRealTimeLocation}
                    onCheckedChange={setUseRealTimeLocation}
                    className="data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-green-500 data-[state=checked]:to-emerald-500"
                  />
                </div>
                <span
                  className={`text-xs font-bold transition-all duration-300 ${useRealTimeLocation
                    ? 'text-green-500 scale-110 drop-shadow-[0_0_8px_rgba(34,197,94,0.6)]'
                    : 'text-muted-foreground/50 scale-90'
                    }`}
                >
                  ON
                </span>
              </div>
            </div>

            <div className="flex flex-col items-center justify-center py-6 text-center">
              {isLocationError ? (
                <div className="p-3 rounded-full bg-orange-100 dark:bg-orange-500/10 mb-3">
                  <MapPin className="w-8 h-8 text-orange-500" />
                </div>
              ) : (
                <AlertTriangle className="w-10 h-10 text-warning mb-3" />
              )}
              <p className="text-sm font-medium text-foreground mb-2">
                {isLocationError ? 'Location Access Required' : 'Unable to load weather'}
              </p>
              <p className="text-xs text-muted-foreground mb-4 max-w-xs">
                {error || 'An unknown error occurred while loading weather data.'}
              </p>
              {isLocationError && (
                <p className="text-xs text-muted-foreground/70 mb-4 max-w-xs">
                  💡 Tip: Click the lock/location icon in your browser's address bar to enable location access.
                </p>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={() => window.location.reload()}
                className="gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                Retry
              </Button>
            </div>
          </CardContent>
        </div>
      </Card>
    );
  }

  const isHighRisk = weather.humidity > 70;

  return (
    <Card className="overflow-hidden border-border/50 shadow-soft">
      <div className="gradient-sky p-1">
        <CardContent className="p-4 bg-card/95 backdrop-blur-sm rounded-lg">
          {/* Location Toggle */}
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-border/50">
            <div className="flex items-center gap-2">
              <MapPin className="w-4 h-4 text-muted-foreground" />
              <Label htmlFor="location-toggle" className="text-sm font-medium cursor-pointer">
                Current Location Weather
              </Label>
            </div>
            <div className="flex items-center gap-3 p-2 rounded-full bg-muted/30">
              <span
                className={`text-xs font-bold transition-all duration-300 ${useRealTimeLocation
                  ? 'text-muted-foreground/50 scale-90'
                  : 'text-red-500 scale-110 drop-shadow-[0_0_8px_rgba(239,68,68,0.6)]'
                  }`}
              >
                OFF
              </span>
              <div className={`relative transition-all duration-300 ${useRealTimeLocation
                ? 'drop-shadow-[0_0_12px_rgba(34,197,94,0.8)]'
                : ''
                }`}>
                <Switch
                  id="location-toggle"
                  checked={useRealTimeLocation}
                  onCheckedChange={setUseRealTimeLocation}
                  className="data-[state=checked]:bg-gradient-to-r data-[state=checked]:from-green-500 data-[state=checked]:to-emerald-500"
                />
              </div>
              <span
                className={`text-xs font-bold transition-all duration-300 ${useRealTimeLocation
                  ? 'text-green-500 scale-110 drop-shadow-[0_0_8px_rgba(34,197,94,0.6)]'
                  : 'text-muted-foreground/50 scale-90'
                  }`}
              >
                ON
              </span>
            </div>
          </div>

          {/* Weather Content - Only show when location is enabled */}
          {useRealTimeLocation && (
            <>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                {/* Location & Main Temp */}
                <div className="flex items-center gap-4">
                  <div className="p-3 rounded-2xl bg-sky/20">
                    {weather.condition.includes("Cloud") ? (
                      <Cloud className="w-10 h-10 text-sky" />
                    ) : (
                      <Sun className="w-10 h-10 text-sun" />
                    )}
                  </div>
                  <div>
                    <p className="text-sm text-muted-foreground flex items-center gap-1">
                      <MapPin className="w-3 h-3" />
                      {weather.location}
                    </p>
                    <div className="flex items-baseline gap-1">
                      <span className="text-4xl font-bold text-foreground">{weather.temperature}</span>
                      <span className="text-xl text-muted-foreground">°C</span>
                    </div>
                    <p className="text-sm font-medium text-foreground">{weather.condition}</p>
                  </div>
                </div>

                {/* Weather Stats */}
                <div className="flex gap-4 sm:gap-6">
                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-lg bg-primary/10">
                      <Droplets className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Humidity</p>
                      <p className="text-lg font-semibold">{weather.humidity}%</p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <div className="p-2 rounded-lg bg-muted">
                      <Wind className="w-5 h-5 text-muted-foreground" />
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Wind</p>
                      <p className="text-lg font-semibold">{weather.windSpeed} km/h</p>
                    </div>
                  </div>

                  <div className="hidden sm:flex items-center gap-2">
                    <div className="p-2 rounded-lg bg-accent/20">
                      <ThermometerSun className="w-5 h-5 text-accent" />
                    </div>
                    <div>
                      <p className="text-xs text-muted-foreground">Feels like</p>
                      <p className="text-lg font-semibold">{weather.feelsLike}°C</p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Recommendation Banner */}
              <div className={`mt-4 p-3 rounded-lg flex items-center gap-3 ${isHighRisk ? "bg-warning/10 border border-warning/30" : "bg-success/10 border border-success/30"
                }`}>
                <AlertTriangle className={`w-5 h-5 shrink-0 ${isHighRisk ? "text-warning" : "text-success"}`} />
                <p className={`text-sm font-medium ${isHighRisk ? "text-warning" : "text-success"}`}>
                  {weather.recommendation}
                </p>
              </div>
            </>
          )}
        </CardContent>
      </div>
    </Card>
  );
};
