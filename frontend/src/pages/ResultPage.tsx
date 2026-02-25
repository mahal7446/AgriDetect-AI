import { useLocation, useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  Leaf,
  AlertTriangle,
  Check,
  Shield,
  Pill,
  Share2,
  Download,
  BookOpen,
  Loader2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { useState, useEffect, useRef } from "react";
import { PredictionResponse, getHistory, getScanImageUrl } from "@/lib/api";
import { useChatContext } from "@/contexts/ChatContext";
import { useToast } from "@/hooks/use-toast";

interface DiseaseResult {
  diseaseName: string;
  confidence: number;
  cropName: string;
  severity: "Low" | "Medium" | "High";
  treatments: string[];
  preventions: string[];
  description: string;
  imageUrl: string;
}

// Treatment and prevention data
const getDiseaseInfo = (diseaseName: string, cropName: string): { treatments: string[]; preventions: string[]; description: string } => {
  const diseaseInfo: Record<string, { treatments: string[]; preventions: string[]; description: string }> = {
    "Early Blight": {
      description: "Early blight is a common fungal disease that affects tomato and potato plants. It is caused by Alternaria solani and typically appears on older leaves first.",
      treatments: [
        "Apply copper-based fungicide every 7-10 days",
        "Remove and destroy infected leaves immediately",
        "Use neem oil spray as organic alternative",
        "Ensure proper spacing between plants for air circulation",
      ],
      preventions: [
        "Rotate crops yearly - avoid planting in same spot",
        "Water at the base of plants, not on leaves",
        "Mulch around plants to prevent soil splash",
        "Choose disease-resistant tomato varieties",
        "Maintain proper plant spacing for airflow",
      ],
    },
    "Late Blight": {
      description: "Late blight is a devastating disease caused by Phytophthora infestans. It spreads rapidly in cool, wet conditions and can destroy entire crops.",
      treatments: [
        "Apply fungicides containing chlorothalonil or mancozeb",
        "Remove and destroy all infected plant parts",
        "Improve air circulation around plants",
        "Avoid overhead watering",
      ],
      preventions: [
        "Plant disease-resistant varieties",
        "Space plants adequately for air flow",
        "Water early in the day so foliage dries",
        "Remove volunteer plants and crop debris",
      ],
    },
    "Healthy": {
      description: "Your plant appears to be healthy! Continue monitoring and maintaining good growing conditions.",
      treatments: [
        "Continue regular monitoring",
        "Maintain proper watering schedule",
        "Ensure adequate nutrition",
        "Keep plants well-spaced",
      ],
      preventions: [
        "Practice crop rotation",
        "Use disease-free seeds and plants",
        "Maintain proper plant spacing",
        "Monitor regularly for early signs of disease",
      ],
    },
  };

  const disease = diseaseName.split('_').pop() || diseaseName;

  return diseaseInfo[disease] || {
    description: `${diseaseName} detected in ${cropName}. Consult with agricultural experts for specific treatment recommendations.`,
    treatments: [
      "Consult with agricultural extension services",
      "Identify specific disease characteristics",
      "Apply appropriate fungicides or treatments",
      "Remove severely infected plants if necessary",
    ],
    preventions: [
      "Practice good crop management",
      "Use disease-resistant varieties when available",
      "Maintain proper plant spacing and air circulation",
      "Monitor plants regularly for early detection",
    ],
  };
};

const severityConfig = {
  Low: { color: "bg-success/20 text-success border-success/30", icon: Check },
  Medium: { color: "bg-warning/20 text-warning border-warning/30", icon: AlertTriangle },
  High: { color: "bg-destructive/20 text-destructive border-destructive/30", icon: AlertTriangle },
};

export const ResultPage = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { setContext, triggerChat } = useChatContext();
  const { toast } = useToast();
  const resultRef = useRef<HTMLDivElement>(null);

  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<DiseaseResult | null>(null);

  useEffect(() => {
    const loadResult = async () => {
      const state = location.state;

      // OPTION 1: Navigated from Upload (has full prediction data)
      if (state?.prediction) {
        const pred = state.prediction as PredictionResponse;
        const info = getDiseaseInfo(pred.diseaseName, pred.cropName);
        setResult({
          diseaseName: pred.diseaseName,
          confidence: pred.confidence,
          cropName: pred.cropName,
          severity: pred.severity,
          description: pred.description || info.description,
          treatments: info.treatments,
          preventions: info.preventions,
          imageUrl: state.imageUrl || "/placeholder.svg"
        });
        return;
      }

      // OPTION 2: Navigated from History (has scanId)
      if (state?.scanId) {
        setIsLoading(true);
        try {
          const userStr = localStorage.getItem('user');
          if (userStr) {
            const user = JSON.parse(userStr);
            const response = await getHistory(user.email);
            if (response.success) {
              const scan = response.history.find(h => h.id === state.scanId);
              if (scan) {
                const info = getDiseaseInfo(scan.diseaseName, scan.cropName);
                setResult({
                  diseaseName: scan.diseaseName,
                  confidence: scan.confidence,
                  cropName: scan.cropName,
                  severity: scan.severity,
                  description: info.description,
                  treatments: info.treatments,
                  preventions: info.preventions,
                  imageUrl: getScanImageUrl(scan.imageUrl)
                });
              }
            }
          }
        } catch (error) {
          console.error("Failed to load historical scan:", error);
          toast({
            title: "Error",
            description: "Failed to load scan details",
            variant: "destructive",
          });
        } finally {
          setIsLoading(false);
        }
        return;
      }

      // Fallback: If no state, go back
      navigate("/history");
    };

    loadResult();
  }, [location, navigate, toast]);

  // Update chat context when result is loaded
  const { context: currentContext } = useChatContext();
  useEffect(() => {
    if (result) {
      // Only update if context is different to prevent infinite loops
      if (!currentContext ||
        currentContext.crop !== result.cropName ||
        currentContext.disease !== result.diseaseName ||
        currentContext.imageUrl !== result.imageUrl) {
        setContext({
          crop: result.cropName,
          disease: result.diseaseName,
          confidence: result.confidence,
          severity: result.severity,
          imageUrl: result.imageUrl,
        });
      }
    }
  }, [result, setContext, currentContext]);

  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <Loader2 className="w-12 h-12 animate-spin text-primary" />
        <p className="text-muted-foreground font-medium">Loading report details...</p>
      </div>
    );
  }

  if (!result) return null;

  const SeverityIcon = severityConfig[result.severity]?.icon || AlertTriangle;

  // Share functionality
  const handleShare = async () => {
    const shareText = `🌱 AgriDetect AI Prediction Report\n\n` +
      `🌾 Crop: ${result.cropName}\n` +
      `🔬 Disease: ${result.diseaseName}\n` +
      `📊 Confidence: ${result.confidence}%\n` +
      `⚠️ Severity: ${result.severity}\n\n` +
      `📝 Description: ${result.description}\n\n` +
      `💊 Treatments:\n${result.treatments.map((t, i) => `${i + 1}. ${t}`).join('\n')}\n\n` +
      `🛡️ Preventions:\n${result.preventions.map((p, i) => `${i + 1}. ${p}`).join('\n')}\n\n` +
      `Analyzed by AgriDetect AI`;

    if (navigator.share) {
      try {
        await navigator.share({
          title: `${result.cropName} - ${result.diseaseName} Detection`,
          text: shareText,
        });
        toast({
          title: "Shared successfully!",
          description: "Your prediction report has been shared.",
        });
      } catch (error) {
        if ((error as Error).name !== 'AbortError') {
          await navigator.clipboard.writeText(shareText);
          toast({
            title: "Copied to clipboard!",
            description: "Share text copied. You can paste it anywhere.",
          });
        }
      }
    } else {
      await navigator.clipboard.writeText(shareText);
      toast({
        title: "Copied to clipboard!",
        description: "Share text copied. You can paste it anywhere.",
      });
    }
  };

  // Download functionality
  const handleDownload = async () => {
    const reportText = `AgriDetect AI - Disease Prediction Report
========================================

Date: ${new Date().toLocaleDateString()}
Time: ${new Date().toLocaleTimeString()}

CROP INFORMATION
----------------
Crop: ${result.cropName}
Disease: ${result.diseaseName}
Confidence: ${result.confidence}%
Severity: ${result.severity}

DESCRIPTION
-----------
${result.description}

TREATMENT SUGGESTIONS
---------------------
${result.treatments.map((t, i) => `${i + 1}. ${t}`).join('\n')}

PREVENTIVE MEASURES
-------------------
${result.preventions.map((p, i) => `${i + 1}. ${p}`).join('\n')}

========================================
Generated by AgriDetect AI
`;

    const blob = new Blob([reportText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `AgriDetect_${result.cropName}_${result.diseaseName.replace(/\s+/g, '_')}_Report.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast({
      title: "Report downloaded!",
      description: "Your prediction report has been saved.",
    });
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6 animate-fade-in" ref={resultRef}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <Button variant="ghost" onClick={() => navigate(-1)}>
          <ArrowLeft className="w-4 h-4 mr-2" />
          Back
        </Button>

        <div className="flex gap-2">
          <Button variant="outline" size="icon" onClick={handleShare} title="Share">
            <Share2 className="w-4 h-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={handleDownload} title="Download">
            <Download className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Image & Result Summary */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Image Preview */}
        <Card className="overflow-hidden">
          <div className="aspect-square relative bg-muted">
            <img
              src={result.imageUrl}
              alt="Uploaded plant"
              className="w-full h-full object-cover"
            />
            <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/70 to-transparent">
              <p className="text-white text-sm">Scan Image</p>
            </div>
          </div>
        </Card>

        {/* Result Summary */}
        <Card className="shadow-soft">
          <CardHeader className="pb-3">
            <div className="flex items-center gap-2 mb-2">
              <Leaf className="w-5 h-5 text-primary" />
              <span className="text-sm text-muted-foreground">Crop Detected</span>
            </div>
            <CardTitle className="text-3xl">{result.cropName}</CardTitle>
          </CardHeader>

          <CardContent className="space-y-4">
            {/* Disease Name */}
            <div>
              <p className="text-sm text-muted-foreground mb-1">Disease Identified</p>
              <h2 className={`text-2xl font-bold ${result.diseaseName.toLowerCase().includes('healthy') ? 'text-success' : 'text-destructive'}`}>
                {result.diseaseName}
              </h2>
            </div>

            {/* Confidence */}
            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-muted-foreground">Confidence</span>
                <span className="font-semibold">{result.confidence}%</span>
              </div>
              <Progress value={result.confidence} className="h-2" />
            </div>

            {/* Severity */}
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Severity Level</span>
              <Badge variant="outline" className={`${severityConfig[result.severity]?.color || ''} gap-1`}>
                <SeverityIcon className="w-3 h-3" />
                {result.severity}
              </Badge>
            </div>

            <Separator />

            {/* Description */}
            <div>
              <p className="text-sm text-muted-foreground mb-2">About this detection</p>
              <p className="text-sm">{result.description}</p>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Treatment & Prevention */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* Treatment */}
        <Card className="shadow-soft border-l-4 border-l-accent">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Pill className="w-5 h-5 text-accent" />
              Treatment Suggestions
            </CardTitle>
            <CardDescription>How to treat this disease</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {result.treatments.map((treatment, index) => (
                <li key={index} className="flex items-start gap-3">
                  <div className="w-6 h-6 rounded-full bg-accent/20 flex items-center justify-center shrink-0 mt-0.5">
                    <span className="text-xs font-bold text-accent">{index + 1}</span>
                  </div>
                  <span className="text-sm">{treatment}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>

        {/* Prevention */}
        <Card className="shadow-soft border-l-4 border-l-primary">
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <Shield className="w-5 h-5 text-primary" />
              Preventive Measures
            </CardTitle>
            <CardDescription>How to prevent future infections</CardDescription>
          </CardHeader>
          <CardContent>
            <ul className="space-y-3">
              {result.preventions.map((prevention, index) => (
                <li key={index} className="flex items-start gap-3">
                  <Check className="w-5 h-5 text-primary shrink-0 mt-0.5" />
                  <span className="text-sm">{prevention}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </Card>
      </div>

      {/* Action Buttons */}
      <Card className="bg-primary/5 border-primary/20">
        <CardContent className="p-4 sm:p-6">
          <div className="flex flex-col sm:flex-row gap-3 justify-center">
            <Button onClick={() => navigate("/upload")} className="gap-2">
              <Leaf className="w-4 h-4" />
              Scan Another Plant
            </Button>
            <Button
              variant="outline"
              className="gap-2"
              onClick={() => triggerChat(`Tell me more about ${result.diseaseName} in ${result.cropName} plants. What are the detailed treatment options, organic remedies, and when should I seek expert help?`)}
            >
              <BookOpen className="w-4 h-4" />
              Learn More About {result.diseaseName}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
