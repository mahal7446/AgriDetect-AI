import { useState } from "react";
import { Clock, MapPin, MoreVertical, Edit2, Trash2, ShieldAlert, ChevronDown, ChevronUp } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useTranslation } from "react-i18next";
import { getScanImageUrl, CommunityAlert } from "@/lib/api";
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface AlertCardProps {
    alert: CommunityAlert;
    currentUserEmail?: string;
    onDelete?: (id: number) => void;
    onEdit?: (alert: CommunityAlert) => void;
}

const getTimeAgo = (timestamp: string, t: any) => {
    const now = new Date();

    // SQLite timestamps are in format: YYYY-MM-DD HH:MM:SS (UTC)
    // To parse correctly as UTC in JS, we convert ' ' to 'T' and add 'Z'
    const utcTimestamp = timestamp.includes(' ') && !timestamp.includes('Z')
        ? timestamp.replace(' ', 'T') + 'Z'
        : timestamp;

    const created = new Date(utcTimestamp);

    // If invalid date, fallback to original behavior
    if (isNaN(created.getTime())) {
        const fallback = new Date(timestamp);
        if (isNaN(fallback.getTime())) return timestamp;
    }

    const diffMs = now.getTime() - created.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return t("dashboard.communityAlerts.timeAgo.justNow") || "Just now";
    if (diffMins < 60) return t("dashboard.communityAlerts.timeAgo.minutesAgo", { count: diffMins });
    if (diffHours < 24) return t("dashboard.communityAlerts.timeAgo.hoursAgo", { count: diffHours });
    return t("dashboard.communityAlerts.timeAgo.daysAgo", { count: diffDays });
};

export const AlertCard = ({
    alert,
    currentUserEmail,
    onDelete,
    onEdit,
}: AlertCardProps) => {
    const { t } = useTranslation();
    const [showPrevention, setShowPrevention] = useState(false);
    const { id, farmerName, location, diseaseReported, description, preventionMethods, imageUrl, userEmail, createdAt } = alert;
    const isOwner = currentUserEmail && userEmail === currentUserEmail;

    return (
        <div className="hover:scale-[1.01] transition-transform duration-300">
            <Card className="overflow-hidden border-emerald-200/50 hover:border-emerald-400 transition-all hover:shadow-lg bg-gradient-to-br from-background to-emerald-50/20 dark:to-emerald-950/10">
                <CardContent className="p-4">
                    <div className="flex gap-4">
                        {/* Image Thumbnail */}
                        <div className="w-24 h-24 sm:w-28 sm:h-28 rounded-xl overflow-hidden shrink-0 border-2 border-emerald-100 shadow-sm relative group">
                            <img
                                src={getScanImageUrl(imageUrl)}
                                alt={diseaseReported}
                                className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-110"
                            />
                            <div className="absolute inset-0 bg-black/5 group-hover:bg-transparent transition-colors" />
                        </div>

                        {/* Content */}
                        <div className="flex-1 min-w-0 flex flex-col justify-between">
                            <div className="relative">
                                {/* Actions Menu */}
                                {isOwner && (
                                    <div className="absolute right-0 top-0 z-10">
                                        <DropdownMenu>
                                            <DropdownMenuTrigger asChild>
                                                <Button variant="ghost" size="icon" className="h-8 w-8 text-muted-foreground hover:text-emerald-600">
                                                    <MoreVertical className="w-4 h-4" />
                                                </Button>
                                            </DropdownMenuTrigger>
                                            <DropdownMenuContent align="end">
                                                <DropdownMenuItem onClick={() => onEdit?.(alert)}>
                                                    <Edit2 className="w-4 h-4 mr-2" />
                                                    {t("common.edit")}
                                                </DropdownMenuItem>
                                                <DropdownMenuItem
                                                    className="text-destructive focus:text-destructive"
                                                    onClick={() => onDelete?.(id)}
                                                >
                                                    <Trash2 className="w-4 h-4 mr-2" />
                                                    {t("common.delete")}
                                                </DropdownMenuItem>
                                            </DropdownMenuContent>
                                        </DropdownMenu>
                                    </div>
                                )}

                                <div className="pr-8">
                                    <h3 className="font-bold text-emerald-900 dark:text-emerald-100 flex items-center gap-2 flex-wrap">
                                        {farmerName}
                                        <Badge variant="outline" className="text-[10px] py-0 h-4 bg-emerald-50 dark:bg-emerald-900/30 border-emerald-200 text-emerald-700">
                                            {location}
                                        </Badge>
                                    </h3>
                                    <p className="text-sm mt-1 leading-relaxed">
                                        <span className="text-muted-foreground">{t("dashboard.communityAlerts.reports")}</span>{" "}
                                        <span className="text-red-500 font-bold uppercase text-xs tracking-tight">{diseaseReported}</span>
                                    </p>
                                </div>

                                {description && (
                                    <p className="text-xs text-muted-foreground my-2 line-clamp-2 italic">
                                        "{description}"
                                    </p>
                                )}

                                {preventionMethods && (
                                    <div className="mt-2 mb-3">
                                        <button
                                            onClick={() => setShowPrevention(!showPrevention)}
                                            className="flex items-center gap-1 text-[11px] font-semibold text-emerald-600 hover:text-emerald-700 transition-colors"
                                        >
                                            {showPrevention ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                                            {t("dashboard.communityAlerts.preventionMethods")}
                                        </button>

                                        {showPrevention && (
                                            <div className="mt-2 p-2 bg-emerald-50/50 dark:bg-emerald-950/30 rounded-lg border border-emerald-100 dark:border-emerald-900/50 animate-in slide-in-from-top-1 duration-200">
                                                <p className="text-xs text-emerald-800 dark:text-emerald-200">
                                                    {preventionMethods}
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>

                            {/* Footer */}
                            <div className="flex items-center justify-between gap-2 mt-auto pt-2 border-t border-emerald-50 dark:border-emerald-900/40">
                                <div className="flex items-center gap-1 text-[10px] text-muted-foreground">
                                    <Clock className="w-3 h-3" />
                                    <span>{getTimeAgo(createdAt, t)}</span>
                                </div>

                                <div className="flex items-center gap-1 px-2 py-0.5 rounded-full bg-red-50 dark:bg-red-950/20">
                                    <div className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />
                                    <span className="text-[9px] font-bold text-red-600 dark:text-red-400 uppercase tracking-wider">{t("dashboard.communityAlerts.activeStatus")}</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
};
