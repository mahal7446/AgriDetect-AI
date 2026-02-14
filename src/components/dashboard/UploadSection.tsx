import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Camera, Upload, Image, Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { predictDisease, saveScanToHistory } from "@/lib/api";

export const UploadSection = () => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { toast } = useToast();

  const handleFileSelect = async (file: File) => {
    if (!file.type.startsWith("image/")) {
      toast({
        title: "Invalid file type",
        description: "Please upload an image file (JPG, PNG, etc.)",
        variant: "destructive",
      });
      return;
    }

    // Check file size (10MB limit)
    if (file.size > 10 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Please upload an image smaller than 10MB",
        variant: "destructive",
      });
      return;
    }

    setIsProcessing(true);

    try {
      // Call ML model API
      const predictionResult = await predictDisease(file);

      toast({
        title: "Analysis complete! 🔬",
        description: `${predictionResult.diseaseName} detected with ${predictionResult.confidence.toFixed(1)}% confidence`,
      });

      // Save to history if user is logged in
      const userStr = localStorage.getItem('user');
      if (userStr) {
        try {
          const user = JSON.parse(userStr);
          if (user.email) {
            await saveScanToHistory(user.email, predictionResult, file);
          }
        } catch (e) {
          console.error("Failed to save to history:", e);
        }
      }

      // Navigate to result page with both image and prediction data
      navigate("/result", {
        state: {
          imageUrl: URL.createObjectURL(file),
          prediction: predictionResult
        }
      });
    } catch (error) {
      console.error("Prediction error:", error);
      toast({
        title: "Prediction failed",
        description: error instanceof Error ? error.message : "Failed to analyze image. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  return (
    <Card className="border-2 border-dashed border-primary/30 hover:border-primary/50 transition-colors shadow-soft">
      <CardHeader className="text-center pb-2">
        <div className="mx-auto p-4 rounded-2xl bg-primary/10 w-fit mb-2">
          <Sparkles className="w-8 h-8 text-primary" />
        </div>
        <CardTitle className="text-xl">Upload Plant Image</CardTitle>
        <CardDescription>
          Take a photo or upload an image of your plant to detect diseases
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Drop Zone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          className={`
            relative p-8 rounded-xl border-2 border-dashed transition-all cursor-pointer
            ${dragActive
              ? "border-primary bg-primary/5"
              : "border-border hover:border-primary/50 hover:bg-muted/30"
            }
            ${isProcessing ? "pointer-events-none opacity-50" : ""}
          `}
          onClick={() => fileInputRef.current?.click()}
        >
          {isProcessing ? (
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-12 h-12 text-primary animate-spin" />
              <p className="text-lg font-medium">Analyzing your plant...</p>
              <p className="text-sm text-muted-foreground">AI is detecting diseases</p>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-3">
              <div className="p-4 rounded-full bg-muted">
                <Image className="w-8 h-8 text-muted-foreground" />
              </div>
              <p className="text-lg font-medium">Drop your image here</p>
              <p className="text-sm text-muted-foreground">or click to browse files</p>
            </div>
          )}
        </div>

        {/* Action Buttons */}
        <div className="grid grid-cols-2 gap-3">
          <Button
            variant="outline"
            className="h-14 gap-3 text-base"
            onClick={() => fileInputRef.current?.click()}
            disabled={isProcessing}
          >
            <Upload className="w-5 h-5" />
            <span className="hidden sm:inline">Upload from</span> Gallery
          </Button>

          <Button
            className="h-14 gap-3 text-base shadow-glow"
            onClick={() => cameraInputRef.current?.click()}
            disabled={isProcessing}
          >
            <Camera className="w-5 h-5" />
            <span className="hidden sm:inline">Open</span> Camera
          </Button>
        </div>

        {/* Hidden inputs */}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={handleInputChange}
        />
        <input
          ref={cameraInputRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="hidden"
          onChange={handleInputChange}
        />

        {/* Supported formats */}
        <p className="text-xs text-center text-muted-foreground">
          Supported: JPG, PNG, WEBP • Max size: 10MB
        </p>
      </CardContent>
    </Card>
  );
};
