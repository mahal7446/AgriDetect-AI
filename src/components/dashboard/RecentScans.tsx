import { useNavigate } from "react-router-dom";
import { Clock, ChevronRight, Leaf, Loader2 } from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useState, useEffect } from "react";
import { getHistory, getScanImageUrl, HistoryItem } from "@/lib/api";

const severityColors = {
  Low: "bg-success/20 text-success border-success/30",
  Medium: "bg-warning/20 text-warning border-warning/30",
  High: "bg-destructive/20 text-destructive border-destructive/30",
};

export const RecentScans = () => {
  const navigate = useNavigate();
  const [scans, setScans] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchRecentScans = async () => {
      const userStr = localStorage.getItem('user');
      if (!userStr) {
        setLoading(false);
        return;
      }

      try {
        const user = JSON.parse(userStr);
        if (user.email) {
          const response = await getHistory(user.email, 3); // Get only last 3
          if (response.success) {
            setScans(response.history);
          }
        }
      } catch (error) {
        console.error("Failed to fetch recent scans:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchRecentScans();
  }, []);

  const parseUTCDate = (dateStr: string) => {
    // SQLite CURRENT_TIMESTAMP stores UTC but without timezone indicator.
    // Ensure the date string is treated as UTC by appending 'Z' if needed.
    if (!dateStr.endsWith('Z') && !dateStr.includes('+') && !dateStr.includes('T')) {
      return new Date(dateStr.replace(' ', 'T') + 'Z');
    }
    return new Date(dateStr);
  };

  const formatRelativeTime = (dateStr: string) => {
    const date = parseUTCDate(dateStr);
    const now = new Date();
    const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

    if (diffInSeconds < 60) return 'Just now';
    if (diffInSeconds < 3600) return `${Math.floor(diffInSeconds / 60)}m ago`;
    if (diffInSeconds < 86400) return `${Math.floor(diffInSeconds / 3600)}h ago`;
    if (diffInSeconds < 604800) return `${Math.floor(diffInSeconds / 86400)}d ago`;

    return date.toLocaleDateString();
  };

  return (
    <Card className="shadow-soft">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="text-lg flex items-center gap-2">
            <Clock className="w-5 h-5 text-primary" />
            Recent Scans
          </CardTitle>
          <CardDescription>Your latest plant analyses</CardDescription>
        </div>
        <Button variant="ghost" size="sm" onClick={() => navigate("/history")}>
          View All
          <ChevronRight className="w-4 h-4 ml-1" />
        </Button>
      </CardHeader>

      <CardContent className="space-y-3">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="w-8 h-8 animate-spin text-primary" />
          </div>
        ) : scans.length > 0 ? (
          scans.map((scan) => (
            <div
              key={scan.id}
              className="flex items-center gap-3 p-3 rounded-lg bg-muted/30 hover:bg-muted/50 transition-colors cursor-pointer group"
              onClick={() => navigate("/result", { state: { scanId: scan.id } })}
            >
              <div className="w-12 h-12 rounded-lg bg-muted overflow-hidden shrink-0 border border-border/50">
                <img
                  src={getScanImageUrl(scan.imageUrl)}
                  alt={scan.cropName}
                  className="w-full h-full object-cover"
                />
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <Leaf className="w-4 h-4 text-primary shrink-0" />
                  <span className="font-medium truncate">{scan.cropName}</span>
                </div>
                <p className="text-sm text-muted-foreground truncate">{scan.diseaseName}</p>
              </div>

              <div className="flex flex-col items-end gap-1 shrink-0">
                <Badge variant="outline" className={severityColors[scan.severity] || severityColors.Medium}>
                  {scan.severity}
                </Badge>
                <span className="text-xs text-muted-foreground">{formatRelativeTime(scan.scanDate)}</span>
              </div>

              <ChevronRight className="w-4 h-4 text-muted-foreground group-hover:text-foreground transition-colors shrink-0" />
            </div>
          ))
        ) : (
          <div className="text-center py-8 text-muted-foreground">
            <Leaf className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p>No scans yet. Upload your first plant image!</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};
