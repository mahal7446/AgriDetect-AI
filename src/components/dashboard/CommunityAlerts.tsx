import { useState, useEffect, useCallback } from "react";
import { Plus, Users, Bell, RefreshCw } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { AlertCard } from "./AlertCard";
import { AlertSubmissionForm } from "./AlertSubmissionForm";
import { useTranslation } from "react-i18next";
import {
    getAlertsByLocation,
    deleteCommunityAlert,
    CommunityAlert
} from "@/lib/api";
import { useToast } from "@/hooks/use-toast";
import { useNotifications } from "@/contexts/NotificationContext";

export const CommunityAlerts = () => {
    const { t } = useTranslation();
    const { toast } = useToast();
    const { newAlertsCount, resetNewAlertsCount, refreshAlerts } = useNotifications();
    const [alerts, setAlerts] = useState<CommunityAlert[]>([]);
    const [isFormOpen, setIsFormOpen] = useState(false);
    const [isLoading, setIsLoading] = useState(true);
    const [editingAlert, setEditingAlert] = useState<CommunityAlert | null>(null);

    // Get user data from localStorage
    const userString = localStorage.getItem("user");
    const user = userString ? JSON.parse(userString) : null;
    const userEmail = user?.email;

    const fetchAlerts = useCallback(async (showLoading = true) => {
        if (!userEmail) return;

        if (showLoading) setIsLoading(true);
        try {
            const response = await getAlertsByLocation(userEmail, 10);
            if (response.success) {
                setAlerts(response.alerts);
                resetNewAlertsCount();
            }
        } catch (error) {
            console.error("Failed to fetch alerts:", error);
        } finally {
            if (showLoading) setIsLoading(false);
        }
    }, [userEmail, resetNewAlertsCount]);

    // Initial fetch
    useEffect(() => {
        fetchAlerts();
    }, [fetchAlerts]);

    const handleRefresh = async () => {
        await refreshAlerts();
        fetchAlerts(false);
    };

    const handleDeleteAlert = async (id: number) => {
        if (!userEmail) return;

        if (!window.confirm(t("dashboard.communityAlerts.confirmDelete"))) return;

        try {
            const response = await deleteCommunityAlert(id, userEmail);
            if (response.success) {
                toast({
                    title: t("dashboard.communityAlerts.alertDeleted"),
                    description: t("dashboard.communityAlerts.alertDeletedDesc"),
                });
                fetchAlerts(false);
            } else {
                throw new Error(response.error);
            }
        } catch (error) {
            toast({
                title: t("common.error"),
                description: error instanceof Error ? error.message : "Failed to delete alert",
                variant: "destructive",
            });
        }
    };

    const handleEditAlert = (alert: CommunityAlert) => {
        setEditingAlert(alert);
        setIsFormOpen(true);
    };

    const handleSubmitSuccess = () => {
        handleRefresh();
        setEditingAlert(null);
    };

    const handleFormClose = () => {
        setIsFormOpen(false);
        setEditingAlert(null);
    };

    return (
        <div className="animate-in fade-in slide-in-from-bottom-2 duration-500">
            <Card className="shadow-soft overflow-hidden border-emerald-200/50 bg-gradient-to-br from-emerald-50/30 to-transparent dark:from-emerald-950/20 dark:to-transparent">
                <CardHeader className="pb-3 pt-4 border-b border-emerald-100 dark:border-emerald-900/50 mb-3">
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                        <div className="flex items-start gap-3">
                            <div className="p-2.5 rounded-xl bg-emerald-100 dark:bg-emerald-900/50 text-emerald-600">
                                <Users className="w-5 h-5" />
                            </div>
                            <div>
                                <CardTitle className="text-lg font-bold">
                                    {t("dashboard.communityAlerts.title")}
                                </CardTitle>
                                <CardDescription className="text-xs">
                                    {t("dashboard.communityAlerts.startSharing")}
                                </CardDescription>
                            </div>
                        </div>

                        <div className="flex items-center gap-2">
                            {newAlertsCount > 0 && (
                                <Button
                                    onClick={handleRefresh}
                                    size="sm"
                                    variant="outline"
                                    className="bg-emerald-50 border-emerald-200 text-emerald-700 hover:bg-emerald-100 animate-pulse"
                                >
                                    <Bell className="w-3.5 h-3.5 mr-1.5" />
                                    {newAlertsCount} {t("dashboard.communityAlerts.newAlerts")}
                                </Button>
                            )}

                            <Button
                                onClick={() => setIsFormOpen(true)}
                                size="sm"
                                className="bg-emerald-600 hover:bg-emerald-700 text-white shadow-md shadow-emerald-600/20"
                            >
                                <Plus className="w-4 h-4 mr-1" />
                                {t("dashboard.communityAlerts.submitAlert")}
                            </Button>
                        </div>
                    </div>
                </CardHeader>

                <CardContent className="space-y-4">
                    {isLoading ? (
                        <div className="text-center py-12 flex flex-col items-center gap-3">
                            <RefreshCw className="w-8 h-8 text-emerald-600 animate-spin opacity-40" />
                            <p className="text-sm text-muted-foreground">{t("common.loading")}</p>
                        </div>
                    ) : alerts.length === 0 ? (
                        <div className="text-center py-12 text-muted-foreground bg-emerald-50/30 dark:bg-emerald-950/20 rounded-2xl border-2 border-dashed border-emerald-100 dark:border-emerald-900/50">
                            <Users className="w-12 h-12 mx-auto mb-3 opacity-20" />
                            <p className="text-sm font-semibold">{t("dashboard.communityAlerts.noAlerts")}</p>
                            <p className="text-xs mt-1 px-4">{t("dashboard.communityAlerts.noAlertsDesc")}</p>
                        </div>
                    ) : (
                        <div className="grid gap-4 sm:grid-cols-1 lg:grid-cols-2">
                            {alerts.map((alert) => (
                                <AlertCard
                                    key={alert.id}
                                    alert={alert}
                                    currentUserEmail={userEmail}
                                    onDelete={handleDeleteAlert}
                                    onEdit={handleEditAlert}
                                />
                            ))}
                        </div>
                    )}
                </CardContent>
            </Card>

            {/* Alert Submission/Edit Form Modal */}
            <AlertSubmissionForm
                isOpen={isFormOpen}
                onClose={handleFormClose}
                onSubmitSuccess={handleSubmitSuccess}
                userFullName={user?.fullName}
                userEmail={userEmail}
                initialData={editingAlert}
            />
        </div>
    );
};
