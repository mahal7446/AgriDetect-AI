import { useState, useEffect } from "react";
import { useTranslation } from "react-i18next";
import {
    BarChart, Bar, LineChart, Line, Area, XAxis, YAxis, CartesianGrid,
    Tooltip, ResponsiveContainer, Legend, Cell
} from "recharts";
import {
    Leaf, TrendingUp, Shield, AlertTriangle,
    ChevronRight, Calendar, Info, BarChart3, Activity, Sparkles
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
    Table, TableBody, TableCell, TableHead,
    TableHeader, TableRow
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import {
    getAnalyticsSummary,
    getAnalyticsCharts,
    getAnalyticsReports,
    AnalyticsSummary,
    AnalyticsCharts,
    AnalyticsReport
} from "@/lib/api";

export const AnalyticsPage = () => {
    const { t } = useTranslation();
    const [loading, setLoading] = useState(true);
    const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
    const [charts, setCharts] = useState<AnalyticsCharts | null>(null);
    const [reports, setReports] = useState<AnalyticsReport[]>([]);

    useEffect(() => {
        const savedUser = localStorage.getItem('user');
        if (savedUser) {
            const user = JSON.parse(savedUser);
            fetchData(user.email);
        }
    }, []);

    const fetchData = async (email: string) => {
        setLoading(true);
        try {
            const [summaryRes, chartsRes, reportsRes] = await Promise.all([
                getAnalyticsSummary(email),
                getAnalyticsCharts(email),
                getAnalyticsReports(email)
            ]);

            if (summaryRes.success) setSummary(summaryRes.summary);
            if (chartsRes.success) setCharts(chartsRes.charts);
            if (reportsRes.success) setReports(reportsRes.reports);
        } catch (error) {
            console.error("Error fetching analytics data:", error);
        } finally {
            setLoading(false);
        }
    };

    const getRiskBadgeVariant = (risk: string) => {
        switch (risk.toLowerCase()) {
            case 'safe': return 'secondary';
            case 'medium': return 'outline';
            case 'high':
            case 'high risk': return 'destructive';
            default: return 'default';
        }
    };

    const COLORS = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6'];

    if (loading) {
        return (
            <div className="flex items-center justify-center min-h-[60vh]">
                <div className="flex flex-col items-center gap-4">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary"></div>
                    <p className="text-muted-foreground animate-pulse">{t('common.loading')}</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-8 animate-in fade-in duration-700">
            {/* Header section */}
            <div className="flex flex-col md:flex-row md:items-end justify-between gap-4">
                <div className="space-y-1">
                    <h1 className="text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
                        <BarChart3 className="w-8 h-8 text-primary" />
                        {t('analytics.title')}
                    </h1>
                    <p className="text-muted-foreground">
                        {t('analytics.subtitle')}
                    </p>
                </div>
                <Badge variant="outline" className="w-fit py-1 px-3 bg-primary/5 border-primary/20 text-primary flex items-center gap-2">
                    <Activity className="w-3 h-3 animate-pulse" />
                    Real-time insights active
                </Badge>
            </div>

            {/* Summary Cards - Descriptive Analytics */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                {[
                    {
                        label: t('analytics.summary.totalScans'),
                        value: summary?.totalScans || 0,
                        icon: Leaf,
                        color: "text-emerald-500",
                        bg: "bg-emerald-500/10",
                        desc: "Past activity"
                    },
                    {
                        label: t('analytics.summary.avgHealth'),
                        value: `${summary?.averageHealth || 0}%`,
                        icon: Shield,
                        color: "text-blue-500",
                        bg: "bg-blue-500/10",
                        desc: "Overall vitality"
                    },
                    {
                        label: t('analytics.summary.alerts'),
                        value: summary?.diseaseAlerts || 0,
                        icon: AlertTriangle,
                        color: "text-amber-500",
                        bg: "bg-amber-500/10",
                        desc: "Action required"
                    },
                    {
                        label: t('analytics.summary.yield'),
                        value: `${summary?.estimatedYield || 0}%`,
                        icon: TrendingUp,
                        color: "text-purple-500",
                        bg: "bg-purple-500/10",
                        desc: "Forecasted outcome"
                    },
                ].map((item, i) => (
                    <Card key={i} className="border-border/50 hover:border-primary/50 transition-all duration-300 shadow-soft group">
                        <CardContent className="p-6">
                            <div className="flex items-start justify-between">
                                <div className="space-y-1">
                                    <p className="text-sm font-medium text-muted-foreground">{item.label}</p>
                                    <p className={`text-3xl font-bold tracking-tighter ${item.color}`}>
                                        {item.value}
                                    </p>
                                    <p className="text-[10px] text-muted-foreground/60 uppercase tracking-wider font-semibold">
                                        {item.desc}
                                    </p>
                                </div>
                                <div className={`p-3 rounded-xl ${item.bg} group-hover:scale-110 transition-transform duration-300`}>
                                    <item.icon className={`w-5 h-5 ${item.color}`} />
                                </div>
                            </div>
                        </CardContent>
                    </Card>
                ))}
            </div>

            {/* Charts - Predictive Analytics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Yield Forecast */}
                <Card className="border-border/50 shadow-soft overflow-hidden">
                    <CardHeader className="bg-muted/30 pb-4">
                        <div className="flex items-center justify-between">
                            <div>
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <TrendingUp className="w-5 h-5 text-primary" />
                                    {t('analytics.charts.yieldForecast')}
                                </CardTitle>
                                <CardDescription className="text-xs">
                                    Based on historical health metrics per crop
                                </CardDescription>
                            </div>
                            <Badge variant="secondary" className="bg-primary/10 text-primary border-none">Predictive</Badge>
                        </div>
                    </CardHeader>
                    <CardContent className="p-6">
                        <div className="h-[300px] w-full mt-4">
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={charts?.yieldForecast}>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--muted-foreground) / 0.1)" />
                                    <XAxis
                                        dataKey="crop"
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                                    />
                                    <YAxis
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                                        domain={[0, 100]}
                                    />
                                    <Tooltip
                                        cursor={{ fill: 'hsl(var(--muted) / 0.5)' }}
                                        contentStyle={{
                                            backgroundColor: 'hsl(var(--card))',
                                            borderColor: 'hsl(var(--border))',
                                            borderRadius: '8px',
                                            boxShadow: '0 4px 12px rgba(0,0,0,0.1)'
                                        }}
                                    />
                                    <Bar dataKey="yield" radius={[6, 6, 0, 0]}>
                                        {charts?.yieldForecast.map((_entry, index) => (
                                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} fillOpacity={0.8} />
                                        ))}
                                    </Bar>
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>

                {/* Disease Trends */}
                <Card className="border-border/50 shadow-soft overflow-hidden">
                    <CardHeader className="bg-muted/30 pb-4">
                        <div className="flex items-center justify-between">
                            <div>
                                <CardTitle className="text-lg flex items-center gap-2">
                                    <Activity className="w-5 h-5 text-blue-500" />
                                    {t('analytics.charts.diseaseTrends')}
                                </CardTitle>
                                <CardDescription className="text-xs">
                                    Monthly progression of crop health
                                </CardDescription>
                            </div>
                            <Badge variant="secondary" className="bg-blue-500/10 text-blue-500 border-none">Analysis</Badge>
                        </div>
                    </CardHeader>
                    <CardContent className="p-6">
                        <div className="h-[300px] w-full mt-4">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={charts?.diseaseTrends}>
                                    <defs>
                                        <linearGradient id="healthyGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="diseasedGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--muted-foreground) / 0.1)" />
                                    <XAxis
                                        dataKey="month"
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                                    />
                                    <YAxis
                                        axisLine={false}
                                        tickLine={false}
                                        tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 12 }}
                                        allowDecimals={false}
                                    />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: 'hsl(var(--card))',
                                            borderColor: 'hsl(var(--border))',
                                            borderRadius: '8px',
                                            boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
                                        }}
                                    />
                                    <Legend verticalAlign="top" align="right" />
                                    <Area
                                        type="monotone"
                                        dataKey="healthy"
                                        fill="url(#healthyGradient)"
                                        stroke="none"
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="diseased"
                                        fill="url(#diseasedGradient)"
                                        stroke="none"
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="healthy"
                                        name={t('analytics.charts.healthy')}
                                        stroke="#10b981"
                                        strokeWidth={3}
                                        dot={{ fill: '#10b981', r: 5, strokeWidth: 2, stroke: '#fff' }}
                                        activeDot={{ r: 7, strokeWidth: 0 }}
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="diseased"
                                        name={t('analytics.charts.diseased')}
                                        stroke="#ef4444"
                                        strokeWidth={3}
                                        dot={{ fill: '#ef4444', r: 5, strokeWidth: 2, stroke: '#fff' }}
                                        activeDot={{ r: 7, strokeWidth: 0 }}
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </CardContent>
                </Card>
            </div>

            {/* Reports Table - Prescriptive Analytics */}
            <Card className="border-border/50 shadow-soft overflow-hidden">
                <CardHeader className="pb-2">
                    <CardTitle className="text-xl flex items-center gap-2">
                        <ChevronRight className="w-5 h-5 text-primary" />
                        {t('analytics.reports.title')}
                    </CardTitle>
                    <CardDescription>
                        High-risk cases requiring immediate attention
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="rounded-md border border-border/50 overflow-hidden">
                        <Table>
                            <TableHeader className="bg-muted/50">
                                <TableRow>
                                    <TableHead className="w-[100px] font-bold">{t('analytics.reports.scanId')}</TableHead>
                                    <TableHead className="font-bold">{t('analytics.reports.date')}</TableHead>
                                    <TableHead className="font-bold">{t('analytics.reports.crop')}</TableHead>
                                    <TableHead className="font-bold">{t('analytics.reports.diagnosis')}</TableHead>
                                    <TableHead className="text-right font-bold">{t('analytics.reports.risk')}</TableHead>
                                </TableRow>
                            </TableHeader>
                            <TableBody>
                                {reports.length > 0 ? (
                                    reports.map((report) => (
                                        <TableRow key={report.id} className="hover:bg-muted/30 transition-colors">
                                            <TableCell className="font-medium text-muted-foreground">#{report.id}</TableCell>
                                            <TableCell className="flex items-center gap-2 text-sm">
                                                <Calendar className="w-3 h-3 text-muted-foreground" />
                                                {new Date(report.date).toLocaleDateString()}
                                            </TableCell>
                                            <TableCell className="font-medium">{report.cropType}</TableCell>
                                            <TableCell>
                                                <span className={report.diagnosis.toLowerCase().includes('healthy') ? "text-emerald-500" : "text-amber-500 font-medium"}>
                                                    {report.diagnosis}
                                                </span>
                                            </TableCell>
                                            <TableCell className="text-right">
                                                <Badge variant={getRiskBadgeVariant(report.riskLevel)} className="font-bold">
                                                    {report.riskLevel}
                                                </Badge>
                                            </TableCell>
                                        </TableRow>
                                    ))
                                ) : (
                                    <TableRow>
                                        <TableCell colSpan={5} className="h-24 text-center text-muted-foreground">
                                            No reports available. Start scanning to see analytics.
                                        </TableCell>
                                    </TableRow>
                                )}
                            </TableBody>
                        </Table>
                    </div>

                    {/* AI Recommendation Placeholder */}
                    <div className="mt-6 p-4 rounded-lg bg-primary/5 border border-primary/10 flex items-start gap-3">
                        <div className="p-2 rounded-full bg-primary/20">
                            <Sparkles className="w-4 h-4 text-primary" />
                        </div>
                        <div className="space-y-1">
                            <h4 className="text-sm font-semibold flex items-center gap-2">
                                Prescriptive Insight
                                <Badge className="bg-primary/20 text-primary border-none text-[10px] h-4">AI ENABLED</Badge>
                            </h4>
                            <p className="text-xs text-muted-foreground">
                                Our AI suggests focusing on **Potato** crops this week as disease trends indicate a 15% increase in your district. Check local community alerts for preventative measures.
                            </p>
                        </div>
                    </div>
                </CardContent>
            </Card>

            {/* Disclaimer */}
            <div className="flex items-center gap-2 text-[10px] text-muted-foreground/50 justify-center">
                <Info className="w-3 h-3" />
                Analytics are estimates based on your scan data and should be verified with field inspection.
            </div>
        </div>
    );
};
