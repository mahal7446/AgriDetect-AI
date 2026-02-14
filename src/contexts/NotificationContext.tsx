import React, { createContext, useContext, useState, useEffect, useCallback, useRef } from "react";
import { getNewAlertsCount, getAlertsByLocation, getNotificationPreference, CommunityAlert } from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { useTranslation } from "react-i18next";
import { Bell } from "lucide-react";

interface NotificationContextType {
    newAlertsCount: number;
    lastSeenId: number;
    resetNewAlertsCount: () => void;
    refreshAlerts: () => Promise<void>;
}

const NotificationContext = createContext<NotificationContextType | undefined>(undefined);

export const NotificationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const { toast } = useToast();
    const { t } = useTranslation();
    const [newAlertsCount, setNewAlertsCount] = useState(0);
    const [lastSeenId, setLastSeenId] = useState(0);
    const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

    // Get user data from localStorage
    const userString = localStorage.getItem("user");
    const user = userString ? JSON.parse(userString) : null;
    const userEmail = user?.email;

    const fetchInitialState = useCallback(async () => {
        if (!userEmail) return;

        try {
            // 1. Check if notifications are enabled
            const prefRes = await getNotificationPreference(userEmail);
            if (prefRes.success && !prefRes.enabled) {
                console.log("Notifications disabled for user");
                return;
            }

            // 2. Get the latest alerts to set the lastSeenId
            const response = await getAlertsByLocation(userEmail, 1);
            if (response.success && response.alerts.length > 0) {
                const maxId = Math.max(...response.alerts.map((a: CommunityAlert) => a.id));
                setLastSeenId(maxId);
            }
        } catch (error) {
            console.error("Failed to initialize notification state:", error);
        }
    }, [userEmail]);

    const pollForAlerts = useCallback(async () => {
        if (!userEmail || lastSeenId === 0) return;

        try {
            const response = await getNewAlertsCount(userEmail, lastSeenId);
            if (response.success && response.count > 0) {
                setNewAlertsCount(prev => prev + response.count);

                // Show a toast notification
                toast({
                    title: t("dashboard.communityAlerts.newAlertsTitle") || "New Disease Alert!",
                    description: t("dashboard.communityAlerts.newAlertsMessage", { count: response.count }) ||
                        `There are ${response.count} new reports in your area.`,
                    action: (
                        <div className="flex items-center gap-2">
                            <Bell className="w-4 h-4 text-emerald-600" />
                        </div>
                    ),
                });
            }
        } catch (error) {
            console.error("Polling error:", error);
        }
    }, [userEmail, lastSeenId, toast, t]);

    useEffect(() => {
        fetchInitialState();
    }, [fetchInitialState]);

    useEffect(() => {
        if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);

        if (userEmail && lastSeenId > 0) {
            pollIntervalRef.current = setInterval(pollForAlerts, 30000); // 30 seconds
        }

        return () => {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
        };
    }, [userEmail, lastSeenId, pollForAlerts]);

    const resetNewAlertsCount = () => {
        setNewAlertsCount(0);
    };

    const refreshAlerts = async () => {
        await fetchInitialState();
        setNewAlertsCount(0);
    };

    return (
        <NotificationContext.Provider value={{ newAlertsCount, lastSeenId, resetNewAlertsCount, refreshAlerts }}>
            {children}
        </NotificationContext.Provider>
    );
};

export const useNotifications = () => {
    const context = useContext(NotificationContext);
    if (context === undefined) {
        throw new Error("useNotifications must be used within a NotificationProvider");
    }
    return context;
};
