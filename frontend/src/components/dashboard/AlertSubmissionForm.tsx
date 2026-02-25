import { useState, useRef, useEffect } from "react";
import { X, Upload, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import { useTranslation } from "react-i18next";
import { submitCommunityAlert, updateCommunityAlert, CommunityAlert, getScanImageUrl } from "@/lib/api";
import { LocationSelector } from "@/components/profile/LocationSelector";

interface AlertSubmissionFormProps {
    isOpen: boolean;
    onClose: () => void;
    onSubmitSuccess: () => void;
    userFullName?: string;
    userEmail?: string;
    initialData?: CommunityAlert | null;
}

export const AlertSubmissionForm = ({
    isOpen,
    onClose,
    onSubmitSuccess,
    userFullName,
    userEmail,
    initialData,
}: AlertSubmissionFormProps) => {
    const { t } = useTranslation();
    const { toast } = useToast();
    const fileInputRef = useRef<HTMLInputElement>(null);
    const isEditMode = !!initialData;

    const [formData, setFormData] = useState({
        farmerName: userFullName || "",
        location: "",
        diseaseReported: "",
        description: "",
        preventionMethods: "",
    });
    const [imageFile, setImageFile] = useState<File | null>(null);
    const [imagePreview, setImagePreview] = useState<string | null>(null);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Load initial data for edit mode
    useEffect(() => {
        if (initialData) {
            setFormData({
                farmerName: initialData.farmerName,
                location: initialData.location,
                diseaseReported: initialData.diseaseReported,
                description: initialData.description || "",
                preventionMethods: initialData.preventionMethods || "",
            });
            if (initialData.imageUrl) {
                setImagePreview(getScanImageUrl(initialData.imageUrl));
            }
        } else {
            setFormData({
                farmerName: userFullName || "",
                location: "",
                diseaseReported: "",
                description: "",
                preventionMethods: "",
            });
            setImagePreview(null);
            setImageFile(null);
        }
    }, [initialData, userFullName, isOpen]);

    const handleImageChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (!file) return;

        // Validate file size (5MB max)
        if (file.size > 5 * 1024 * 1024) {
            toast({
                title: t("profile.fileTooLarge"),
                description: t("profile.selectSmaller"),
                variant: "destructive",
            });
            return;
        }

        // Validate file type
        if (!file.type.startsWith("image/")) {
            toast({
                title: t("profile.invalidFile"),
                description: t("profile.selectImage"),
                variant: "destructive",
            });
            return;
        }

        setImageFile(file);

        // Create preview
        const reader = new FileReader();
        reader.onloadend = () => {
            setImagePreview(reader.result as string);
        };
        reader.readAsDataURL(file);
    };

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        if (!formData.farmerName || !formData.location || !formData.diseaseReported) {
            toast({
                title: t("common.error"),
                description: "Please fill in all required fields",
                variant: "destructive",
            });
            return;
        }

        setIsSubmitting(true);

        try {
            const submitFormData = new FormData();
            submitFormData.append("farmerName", formData.farmerName);
            submitFormData.append("location", formData.location);
            submitFormData.append("diseaseReported", formData.diseaseReported);
            submitFormData.append("description", formData.description);
            submitFormData.append("preventionMethods", formData.preventionMethods);

            if (userEmail) {
                submitFormData.append("userEmail", userEmail);
            }
            if (imageFile) {
                submitFormData.append("image", imageFile);
            }

            let response;
            if (isEditMode && initialData) {
                response = await updateCommunityAlert(initialData.id, submitFormData);
            } else {
                response = await submitCommunityAlert(submitFormData);
            }

            if (!response.success) {
                throw new Error(response.error || "Failed to submit alert");
            }

            toast({
                title: isEditMode
                    ? t("dashboard.communityAlerts.alertUpdated")
                    : t("dashboard.communityAlerts.alertSubmitted"),
                description: isEditMode
                    ? t("dashboard.communityAlerts.alertUpdatedDesc")
                    : t("dashboard.communityAlerts.alertSubmittedDesc"),
            });

            // Reset form (if not edit mode)
            if (!isEditMode) {
                setFormData({
                    farmerName: userFullName || "",
                    location: "",
                    diseaseReported: "",
                    description: "",
                    preventionMethods: "",
                });
                setImageFile(null);
                setImagePreview(null);
            }

            onSubmitSuccess();
            onClose();

        } catch (error) {
            console.error("Submit alert error:", error);
            toast({
                title: t("dashboard.communityAlerts.submitFailed"),
                description: error instanceof Error ? error.message : "Unknown error",
                variant: "destructive",
            });
        } finally {
            setIsSubmitting(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm">
            <div className="bg-background rounded-lg shadow-2xl w-full max-w-md max-h-[90vh] overflow-y-auto animate-in fade-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="sticky top-0 bg-background border-b p-4 flex items-center justify-between z-10">
                    <h2 className="text-xl font-semibold">
                        {isEditMode
                            ? t("dashboard.communityAlerts.editAlert")
                            : t("dashboard.communityAlerts.submitAlert")}
                    </h2>
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={onClose}
                        disabled={isSubmitting}
                    >
                        <X className="w-5 h-5" />
                    </Button>
                </div>

                {/* Form */}
                <form onSubmit={handleSubmit} className="p-4 space-y-4">
                    {/* Farmer Name */}
                    <div>
                        <label className="text-sm font-medium mb-1.5 block">
                            {t("dashboard.communityAlerts.farmerName")} *
                        </label>
                        <Input
                            value={formData.farmerName}
                            onChange={(e) =>
                                setFormData({ ...formData, farmerName: e.target.value })
                            }
                            placeholder={t("dashboard.communityAlerts.farmerName")}
                            required
                        />
                    </div>

                    {/* Location Selection */}
                    <div className="space-y-1.5">
                        <label className="text-sm font-medium block">
                            {t("dashboard.communityAlerts.location")} *
                        </label>
                        <LocationSelector
                            value={formData.location}
                            onChange={(val) => setFormData({ ...formData, location: val })}
                        />
                    </div>

                    {/* Disease Name */}
                    <div>
                        <label className="text-sm font-medium mb-1.5 block">
                            {t("dashboard.communityAlerts.disease")} *
                        </label>
                        <Input
                            value={formData.diseaseReported}
                            onChange={(e) =>
                                setFormData({ ...formData, diseaseReported: e.target.value })
                            }
                            placeholder={t("dashboard.communityAlerts.disease")}
                            required
                        />
                    </div>

                    {/* Description */}
                    <div>
                        <label className="text-sm font-medium mb-1.5 block">
                            {t("dashboard.communityAlerts.description")}
                        </label>
                        <Textarea
                            value={formData.description}
                            onChange={(e) =>
                                setFormData({ ...formData, description: e.target.value })
                            }
                            placeholder={t("dashboard.communityAlerts.description")}
                            rows={2}
                        />
                    </div>

                    {/* Prevention Methods */}
                    <div>
                        <div className="flex items-center gap-2 mb-1.5">
                            <label className="text-sm font-medium block">
                                {t("dashboard.communityAlerts.preventionMethods")}
                            </label>
                            <Info className="w-3.5 h-3.5 text-muted-foreground" />
                        </div>
                        <Textarea
                            value={formData.preventionMethods}
                            onChange={(e) =>
                                setFormData({ ...formData, preventionMethods: e.target.value })
                            }
                            placeholder={t("dashboard.communityAlerts.preventionMethodsPlaceholder")}
                            rows={3}
                            className="text-sm"
                        />
                        <p className="text-[10px] text-muted-foreground mt-1">
                            {t("dashboard.communityAlerts.preventionHint")}
                        </p>
                    </div>

                    {/* Image Upload */}
                    <div>
                        <label className="text-sm font-medium mb-1.5 block">
                            {t("dashboard.communityAlerts.uploadPhoto")}
                        </label>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="image/*"
                            onChange={handleImageChange}
                            className="hidden"
                        />

                        {imagePreview ? (
                            <div className="relative w-full h-40 rounded-lg overflow-hidden border-2 border-dashed border-emerald-300">
                                <img
                                    src={imagePreview}
                                    alt="Preview"
                                    className="w-full h-full object-cover"
                                />
                                <Button
                                    type="button"
                                    variant="destructive"
                                    size="icon"
                                    className="absolute top-2 right-2"
                                    onClick={() => {
                                        setImageFile(null);
                                        setImagePreview(null);
                                        if (fileInputRef.current) {
                                            fileInputRef.current.value = "";
                                        }
                                    }}
                                >
                                    <X className="w-4 h-4" />
                                </Button>
                            </div>
                        ) : (
                            <button
                                type="button"
                                onClick={() => fileInputRef.current?.click()}
                                className="w-full h-32 rounded-lg border-2 border-dashed border-emerald-300 hover:border-emerald-400 transition-colors flex flex-col items-center justify-center gap-2 text-muted-foreground hover:text-foreground"
                            >
                                <Upload className="w-8 h-8" />
                                <span className="text-sm">{t("dashboard.communityAlerts.uploadPhoto")}</span>
                            </button>
                        )}
                    </div>

                    {/* Submit Button */}
                    <Button
                        type="submit"
                        className="w-full bg-emerald-600 hover:bg-emerald-700"
                        disabled={isSubmitting}
                    >
                        {isSubmitting
                            ? t("dashboard.communityAlerts.processing")
                            : isEditMode
                                ? t("dashboard.communityAlerts.update")
                                : t("dashboard.communityAlerts.submit")}
                    </Button>
                </form>
            </div>
        </div>
    );
};
