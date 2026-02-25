import { CommunityAlerts } from "@/components/dashboard/CommunityAlerts";
import { useTranslation } from "react-i18next";

export const CommunityPage = () => {
    const { t } = useTranslation();

    return (
        <div className="space-y-6 animate-in fade-in duration-500">
            {/* Page Header */}
            <div className="space-y-1 animate-in slide-in-from-bottom-2 duration-500">
                <h1 className="text-2xl sm:text-3xl font-bold text-foreground">
                    {t("dashboard.communityAlerts.title")}
                </h1>
                <p className="text-muted-foreground">
                    {t("dashboard.communityAlerts.startSharing")}
                </p>
            </div>

            {/* Community Alerts Section */}
            <div className="animate-in slide-in-from-bottom-4 duration-500" style={{ animationDelay: '200ms' }}>
                <CommunityAlerts />
            </div>
        </div>
    );
};
