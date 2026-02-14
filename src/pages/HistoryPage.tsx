import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Search,
  Filter,
  Calendar,
  Leaf,
  ChevronRight,
  Trash2,
  CheckCircle,
  AlertCircle,
  Loader2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { useTranslation } from "react-i18next";
import { useToast } from "@/hooks/use-toast";
import { getHistory, deleteHistoryItem, getScanImageUrl, HistoryItem } from "@/lib/api";

const severityColors = {
  Low: "bg-success/20 text-success border-success/30",
  Medium: "bg-warning/20 text-warning border-warning/30",
  High: "bg-destructive/20 text-destructive border-destructive/30",
};

export const HistoryPage = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { toast } = useToast();

  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [filterSeverity, setFilterSeverity] = useState<string>("all");
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [deleting, setDeleting] = useState(false);

  // Fetch history on mount
  useEffect(() => {
    fetchHistory();
  }, []);

  const fetchHistory = async () => {
    const userStr = localStorage.getItem('user');
    if (!userStr) {
      setLoading(false);
      return;
    }

    try {
      const user = JSON.parse(userStr);
      if (user.email) {
        const response = await getHistory(user.email);
        if (response.success) {
          setHistory(response.history);
        }
      }
    } catch (error) {
      console.error("Failed to fetch history:", error);
      toast({
        title: "Error",
        description: "Failed to load scan history",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!deleteId) return;

    const userStr = localStorage.getItem('user');
    if (!userStr) return;

    setDeleting(true);
    try {
      const user = JSON.parse(userStr);
      const response = await deleteHistoryItem(deleteId, user.email);

      if (response.success) {
        setHistory(prev => prev.filter(item => item.id !== deleteId));
        toast({
          title: "Deleted",
          description: "Scan removed from history",
        });
      } else {
        toast({
          title: "Error",
          description: response.error || "Failed to delete scan",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete scan",
        variant: "destructive",
      });
    } finally {
      setDeleting(false);
      setDeleteId(null);
    }
  };

  const parseUTCDate = (dateStr: string) => {
    // SQLite CURRENT_TIMESTAMP stores UTC but without timezone indicator.
    // Ensure the date string is treated as UTC by appending 'Z' if needed.
    if (!dateStr.endsWith('Z') && !dateStr.includes('+') && !dateStr.includes('T')) {
      return new Date(dateStr.replace(' ', 'T') + 'Z');
    }
    return new Date(dateStr);
  };

  const formatDate = (dateStr: string) => {
    const date = parseUTCDate(dateStr);
    return {
      date: date.toLocaleDateString('en-IN', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
      }),
      time: date.toLocaleTimeString('en-IN', {
        hour: 'numeric',
        minute: '2-digit',
        hour12: true
      })
    };
  };

  const isHealthy = (diseaseName: string) => {
    return diseaseName.toLowerCase().includes('healthy');
  };

  const filteredHistory = history.filter((item) => {
    const matchesSearch =
      item.cropName?.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.diseaseName.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesSeverity = filterSeverity === "all" || item.severity === filterSeverity;
    return matchesSearch && matchesSeverity;
  });

  const healthyCount = history.filter(h => isHealthy(h.diseaseName)).length;
  const diseasedCount = history.filter(h => !isHealthy(h.diseaseName)).length;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="space-y-1">
        <h1 className="text-2xl sm:text-3xl font-bold text-foreground flex items-center gap-3">
          <Calendar className="w-7 h-7 text-primary" />
          {t('history.pageTitle')}
        </h1>
        <p className="text-muted-foreground">
          {t('history.pageSubtitle')}
        </p>
      </div>

      {/* Search & Filter */}
      <Card className="shadow-soft">
        <CardContent className="p-4">
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder={t('history.searchPlaceholder')}
                className="pl-9"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Select value={filterSeverity} onValueChange={setFilterSeverity}>
              <SelectTrigger className="w-full sm:w-[180px]">
                <Filter className="w-4 h-4 mr-2" />
                <SelectValue placeholder={t('history.filterBySeverity')} />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">{t('history.allSeverities')}</SelectItem>
                <SelectItem value="Low">{t('history.low')}</SelectItem>
                <SelectItem value="Medium">{t('history.medium')}</SelectItem>
                <SelectItem value="High">{t('history.high')}</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </CardContent>
      </Card>

      {/* History List */}
      <div className="space-y-3">
        {filteredHistory.length > 0 ? (
          filteredHistory.map((item) => {
            const { date, time } = formatDate(item.scanDate);
            const healthy = isHealthy(item.diseaseName);

            return (
              <Card
                key={item.id}
                className="shadow-soft hover:shadow-glow transition-all group"
              >
                <CardContent className="p-4">
                  <div className="flex items-center gap-4">
                    {/* Image */}
                    <div
                      className="w-16 h-16 sm:w-20 sm:h-20 rounded-xl overflow-hidden bg-muted shrink-0 cursor-pointer"
                      onClick={() => navigate("/result", { state: { scanId: item.id } })}
                    >
                      <img
                        src={getScanImageUrl(item.imageUrl)}
                        alt={item.cropName || "Scan"}
                        className="w-full h-full object-cover"
                      />
                    </div>

                    {/* Info */}
                    <div
                      className="flex-1 min-w-0 cursor-pointer"
                      onClick={() => navigate("/result", { state: { scanId: item.id } })}
                    >
                      <div className="flex items-center gap-2 mb-1">
                        <Leaf className="w-4 h-4 text-primary shrink-0" />
                        <span className="font-semibold text-lg truncate">{item.cropName || "Unknown"}</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        {healthy ? (
                          <span className="flex items-center gap-1 text-success">
                            <CheckCircle className="w-4 h-4" />
                            {t('history.healthy')}
                          </span>
                        ) : (
                          <span className="flex items-center gap-1 text-destructive">
                            <AlertCircle className="w-4 h-4" />
                            {item.diseaseName}
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {date} {t('history.at')} {time}
                      </p>
                    </div>

                    {/* Severity & Actions */}
                    <div className="flex items-center gap-2 shrink-0">
                      <Badge
                        variant="outline"
                        className={`hidden sm:flex ${severityColors[item.severity] || severityColors.Medium}`}
                      >
                        {item.severity || "Medium"}
                      </Badge>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="text-muted-foreground hover:text-destructive"
                        onClick={(e) => {
                          e.stopPropagation();
                          setDeleteId(item.id);
                        }}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                      <ChevronRight
                        className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors cursor-pointer"
                        onClick={() => navigate("/result", { state: { scanId: item.id } })}
                      />
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })
        ) : (
          <Card className="shadow-soft">
            <CardContent className="p-12 text-center">
              <Leaf className="w-16 h-16 mx-auto mb-4 text-muted-foreground/30" />
              <h3 className="text-lg font-semibold mb-2">{t('history.noScansFound')}</h3>
              <p className="text-muted-foreground mb-4">
                {searchQuery || filterSeverity !== "all"
                  ? t('history.adjustSearch')
                  : t('history.uploadFirst')}
              </p>
              <Button onClick={() => navigate("/upload")}>
                {t('history.uploadPlantImage')}
              </Button>
            </CardContent>
          </Card>
        )}
      </div>

      {/* Summary Stats */}
      {history.length > 0 && (
        <Card className="bg-muted/30">
          <CardContent className="p-4">
            <div className="flex flex-wrap justify-center gap-6 text-center">
              <div>
                <p className="text-2xl font-bold text-foreground">{history.length}</p>
                <p className="text-sm text-muted-foreground">{t('history.totalScans')}</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-success">{healthyCount}</p>
                <p className="text-sm text-muted-foreground">{t('history.healthyPlants')}</p>
              </div>
              <div>
                <p className="text-2xl font-bold text-destructive">{diseasedCount}</p>
                <p className="text-sm text-muted-foreground">{t('history.diseasedPlants')}</p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={deleteId !== null} onOpenChange={() => setDeleteId(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Scan?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently remove this scan from your history. This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel disabled={deleting}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleDelete}
              disabled={deleting}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {deleting ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : null}
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};
