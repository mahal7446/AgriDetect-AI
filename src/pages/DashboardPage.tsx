import { useState, useEffect } from "react";
import { WeatherWidget } from "@/components/dashboard/WeatherWidget";
import { UploadSection } from "@/components/dashboard/UploadSection";
import { RecentScans } from "@/components/dashboard/RecentScans";
import { Leaf, TrendingUp, Shield, Users, Sparkles } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { useTranslation } from "react-i18next";
import { API_BASE_URL } from "@/lib/api";

export const DashboardPage = () => {
  const { t } = useTranslation();
  const [user, setUser] = useState<any>(null);
  const [statsData, setStatsData] = useState({
    totalScans: 0,
    diseasesDetected: 0,
    accuracyRate: 0,
    communityFarmers: 0
  });

  useEffect(() => {
    // Get user from localStorage
    const savedUser = localStorage.getItem('user');
    if (savedUser) {
      const parsedUser = JSON.parse(savedUser);
      setUser(parsedUser);
      fetchRealTimeData(parsedUser.email);
    }
  }, []);

  const fetchRealTimeData = async (email: string) => {
    try {
      // Fetch scan stats
      const statsRes = await fetch(`${API_BASE_URL}/api/profile/stats?email=${email}`);
      const statsJson = await statsRes.json();

      // Fetch accuracy
      const accuracyRes = await fetch(`${API_BASE_URL}/api/profile/accuracy?email=${email}`);
      const accuracyJson = await accuracyRes.json();

      // Fetch total users (community farmers)
      const usersRes = await fetch(`${API_BASE_URL}/api/stats/total-users`);
      const usersJson = await usersRes.json();

      if (statsJson.success && accuracyJson.success && usersJson.success) {
        setStatsData({
          totalScans: statsJson.stats.total,
          diseasesDetected: statsJson.stats.diseased,
          accuracyRate: accuracyJson.accuracy,
          communityFarmers: usersJson.total_users
        });
      }
    } catch (error) {
      console.error("Error fetching dashboard data:", error);
    }
  };

  const statCards = [
    {
      label: "Total Scans",
      value: statsData.totalScans.toString(),
      icon: Leaf,
      gradient: "from-emerald-400 to-teal-600",
      shadowColor: "shadow-emerald-500/50"
    },
    {
      label: "Diseases Detected",
      value: statsData.diseasesDetected.toString(),
      icon: Shield,
      gradient: "from-emerald-400 to-teal-600",
      shadowColor: "shadow-emerald-500/50"
    },
    {
      label: "Accuracy Rate",
      value: `${Math.min(statsData.accuracyRate, 99.9).toFixed(1)}%`,
      icon: TrendingUp,
      gradient: "from-emerald-400 to-teal-600",
      shadowColor: "shadow-emerald-500/50"
    },
    {
      label: "Community Farmers",
      value: statsData.communityFarmers >= 1000
        ? `${(statsData.communityFarmers / 1000).toFixed(1)}K`
        : statsData.communityFarmers.toString(),
      icon: Users,
      gradient: "from-emerald-400 to-teal-600",
      shadowColor: "shadow-emerald-500/50"
    },
  ];

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Welcome Section */}
      <div className="space-y-1 animate-in slide-in-from-bottom-2 duration-500">
        <h1 className="text-2xl sm:text-3xl font-bold text-foreground flex items-center gap-2">
          {t('dashboard.welcome')},{" "}
          <span className="bg-gradient-to-r from-primary via-accent to-primary bg-clip-text text-transparent animate-gradient">
            {user?.fullName?.split(' ')[0] || 'Farmer'}!
          </span>{" "}
          🌾
        </h1>
        <p className="text-muted-foreground">
          {t('upload.subtitle')}
        </p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
        {statCards.map((stat, index) => {
          const Icon = stat.icon;
          return (
            <div
              key={stat.label}
              className="hover:-translate-y-2 hover:scale-103 active:scale-98 transition-all duration-200 animate-in slide-in-from-bottom-4"
              style={{ animationDelay: `${index * 100}ms` }}
            >
              <Card className={`relative overflow-hidden border-0 shadow-lg hover:shadow-2xl ${stat.shadowColor} transition-all duration-300`}>
                {/* Gradient Background */}
                <div className={`absolute inset-0 bg-gradient-to-br ${stat.gradient} opacity-90`} />

                {/* Glass effect overlay */}
                <div className="absolute inset-0 bg-white/10 backdrop-blur-sm" />

                <CardContent className="relative p-4 flex items-center gap-3 text-white">
                  <div className="p-3 rounded-xl bg-white/20 backdrop-blur-md hover:rotate-360 transition-transform duration-600">
                    <Icon className="w-6 h-6 text-white drop-shadow-lg" />
                  </div>
                  <div>
                    <p className="text-3xl font-bold drop-shadow-md">
                      {stat.value}
                    </p>
                    <p className="text-sm text-white/90 font-medium">{stat.label}</p>
                  </div>
                </CardContent>

                {/* Shine effect */}
                <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/20 to-transparent -translate-x-full hover:translate-x-full transition-transform duration-600" />
              </Card>
            </div>
          );
        })}
      </div>

      {/* Weather Widget */}
      <div className="animate-in slide-in-from-bottom-4 duration-500" style={{ animationDelay: '400ms' }}>
        <WeatherWidget />
      </div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Upload Section */}
        <div className="animate-in slide-in-from-left duration-500" style={{ animationDelay: '500ms' }}>
          <UploadSection />
        </div>

        {/* Recent Scans */}
        <div className="animate-in slide-in-from-right duration-500" style={{ animationDelay: '600ms' }}>
          <RecentScans />
        </div>
      </div>

      {/* Tips Section */}
      <div className="animate-in slide-in-from-bottom-4 duration-500" style={{ animationDelay: '700ms' }}>
        <Card className="relative overflow-hidden border-primary/30 shadow-lg hover:shadow-xl transition-shadow">
          {/* Gradient background */}
          <div className="absolute inset-0 bg-gradient-to-br from-primary/10 via-accent/5 to-primary/10" />

          <CardContent className="relative p-4 sm:p-6">
            <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-primary" />
              Tips for Best Results
            </h3>
            <ul className="grid sm:grid-cols-2 gap-3 text-sm text-muted-foreground">
              {[
                "Take photos in good natural lighting",
                "Focus on the affected area of the plant",
                "Include both healthy and diseased parts if possible",
                "Avoid blurry or shadowy images"
              ].map((tip, index) => (
                <li
                  key={index}
                  className="flex items-start gap-2 cursor-default hover:translate-x-1 hover:text-primary transition-all"
                >
                  <span className="text-primary text-lg">•</span>
                  {tip}
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
