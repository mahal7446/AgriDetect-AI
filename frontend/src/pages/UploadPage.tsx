import { UploadSection } from "@/components/dashboard/UploadSection";
import { Leaf, CheckCircle } from "lucide-react";
import { Card, CardContent } from "@/components/ui/card";
import { useTranslation } from "react-i18next";



export const UploadPage = () => {
  const { t } = useTranslation();

  const tips = [
    t('upload.tip1'),
    t('upload.tip2'),
    t('upload.tip3'),
    t('upload.tip4'),
    t('upload.tip5'),
  ];

  return (
    <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
      {/* Header */}
      <div className="text-center space-y-2">
        <h1 className="text-2xl sm:text-3xl font-bold text-foreground flex items-center justify-center gap-3">
          <Leaf className="w-7 h-7 text-primary" />
          {t('upload.pageTitle')}
        </h1>
        <p className="text-muted-foreground">
          {t('upload.pageSubtitle')}
        </p>
      </div>

      {/* Upload Section */}
      <UploadSection />

      {/* Tips Card */}
      <Card className="bg-primary/5 border-primary/20">
        <CardContent className="p-6">
          <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
            <Leaf className="w-5 h-5 text-primary" />
            {t('upload.tipsTitle')}
          </h3>
          <ul className="space-y-3">
            {tips.map((tip, index) => (
              <li key={index} className="flex items-start gap-3">
                <CheckCircle className="w-5 h-5 text-primary shrink-0 mt-0.5" />
                <span className="text-sm text-muted-foreground">{tip}</span>
              </li>
            ))}
          </ul>
        </CardContent>
      </Card>

      {/* Supported Crops */}
      <Card className="shadow-soft">
        <CardContent className="p-6">
          <h3 className="font-semibold text-lg mb-4">{t('upload.supportedCrops')}</h3>
          <div className="flex flex-wrap gap-2">
            {["Rice", "Potato", "Corn", "Blackgram", "Tomato", "Cotton", "Wheat", "Pumpkin"].map((crop) => (
              <span
                key={crop}
                className="px-3 py-1 bg-muted rounded-full text-sm font-medium"
              >
                {crop}
              </span>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
};
