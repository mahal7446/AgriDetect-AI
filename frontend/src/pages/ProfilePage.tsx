import { useState, useEffect, useRef } from "react";
import {
  User,
  Mail,
  Phone,
  MapPin,
  Edit2,
  Save,
  Languages,
  Bell,
  Moon,
  LogOut,
  Leaf,
  Camera,
  Loader2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useToast } from "@/hooks/use-toast";
import { useNavigate } from "react-router-dom";
import { getProfile, updateProfile, uploadProfilePicture, getProfilePictureUrl, getNotificationPreference, updateNotificationPreference, getUserStats } from "@/lib/api";
import { useTheme } from "@/contexts/ThemeContext";
import { useTranslation } from "react-i18next";
import { LocationSelector } from "@/components/profile/LocationSelector";

export const ProfilePage = () => {
  const { toast } = useToast();
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { theme, toggleTheme } = useTheme();
  const { t, i18n } = useTranslation();

  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [isUploadingPicture, setIsUploadingPicture] = useState(false);

  const [stats, setStats] = useState({
    total: 0,
    healthy: 0,
    diseased: 0
  });

  const [profile, setProfile] = useState({
    name: "",
    email: "",
    phone: "",
    address: "",
    language: "en",
    profilePictureUrl: "",
    notificationsEnabled: true,
  });

  // Load user data on component mount
  useEffect(() => {
    const loadUserProfile = async () => {
      try {
        // Get user from localStorage
        const userStr = localStorage.getItem('user');
        if (!userStr) {
          toast({
            title: "Not logged in",
            description: "Please log in to view your profile",
            variant: "destructive",
          });
          navigate("/login");
          return;
        }

        const user = JSON.parse(userStr);

        // Fetch full profile, notification preference, and stats from backend
        const [profileRes, notifRes, statsRes] = await Promise.all([
          getProfile(user.email),
          getNotificationPreference(user.email),
          getUserStats(user.email)
        ]);

        if (profileRes.success && profileRes.user) {
          setProfile({
            name: profileRes.user.fullName || "",
            email: profileRes.user.email || "",
            phone: profileRes.user.phone || "",
            address: profileRes.user.address || "",
            language: localStorage.getItem('language') || "en",
            profilePictureUrl: profileRes.user.profilePictureUrl || "",
            notificationsEnabled: notifRes.success ? notifRes.enabled : true,
          });
        } else {
          // Use localStorage data as fallback
          setProfile({
            name: user.fullName || "",
            email: user.email || "",
            phone: user.phone || "",
            address: user.address || "",
            language: localStorage.getItem('language') || "en",
            profilePictureUrl: user.profilePictureUrl || "",
            notificationsEnabled: true,
          });
        }

        if (statsRes.success) {
          setStats(statsRes.stats);
        }
      } catch (error) {
        console.error("Error loading profile:", error);
        toast({
          title: "Error loading profile",
          description: "Could not load your profile data",
          variant: "destructive",
        });
      } finally {
        setIsLoading(false);
      }
    };

    loadUserProfile();
  }, [navigate, toast]);

  const handleNotificationToggle = async (enabled: boolean) => {
    try {
      const response = await updateNotificationPreference(profile.email, enabled);
      if (response.success) {
        setProfile(prev => ({ ...prev, notificationsEnabled: enabled }));
        toast({
          title: enabled ? "Notifications enabled" : "Notifications disabled",
          description: enabled
            ? "You will receive alerts from your nearby areas"
            : "You will no longer receive community alerts",
        });
      }
    } catch (error) {
      toast({
        title: "Preference update failed",
        description: "Could not save notification setting",
        variant: "destructive",
      });
    }
  };

  const handleSave = async () => {
    setIsSaving(true);

    try {
      const response = await updateProfile(profile.email, {
        fullName: profile.name,
        phone: profile.phone,
        address: profile.address,
        email: profile.email,
      });

      if (response.success) {
        setIsEditing(false);

        // Update localStorage
        const userStr = localStorage.getItem('user');
        if (userStr) {
          const user = JSON.parse(userStr);
          localStorage.setItem('user', JSON.stringify({
            ...user,
            fullName: profile.name,
            phone: profile.phone,
            address: profile.address,
          }));
        }

        toast({
          title: "Profile updated! ✓",
          description: "Your changes have been saved successfully",
        });
      } else {
        toast({
          title: "Update failed",
          description: response.error || "Failed to update profile",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Update failed",
        description: "An error occurred while saving",
        variant: "destructive",
      });
    } finally {
      setIsSaving(false);
    }
  };

  const handleProfilePictureClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      toast({
        title: "Invalid file",
        description: "Please select an image file",
        variant: "destructive",
      });
      return;
    }

    // Validate file size (5MB)
    if (file.size > 5 * 1024 * 1024) {
      toast({
        title: "File too large",
        description: "Please select an image smaller than 5MB",
        variant: "destructive",
      });
      return;
    }

    setIsUploadingPicture(true);

    try {
      const response = await uploadProfilePicture(profile.email, file);

      if (response.success) {
        const newProfilePictureUrl = response.profilePictureUrl || "";
        setProfile({ ...profile, profilePictureUrl: newProfilePictureUrl });

        // Update localStorage
        const userStr = localStorage.getItem('user');
        if (userStr) {
          const user = JSON.parse(userStr);
          localStorage.setItem('user', JSON.stringify({
            ...user,
            profilePictureUrl: newProfilePictureUrl,
          }));
        }

        toast({
          title: "Profile picture updated! ✓",
          description: "Your profile picture has been uploaded successfully",
        });
      } else {
        toast({
          title: "Upload failed",
          description: response.error || "Failed to upload profile picture",
          variant: "destructive",
        });
      }
    } catch (error) {
      toast({
        title: "Upload failed",
        description: "An error occurred while uploading",
        variant: "destructive",
      });
    } finally {
      setIsUploadingPicture(false);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('user');
    toast({
      title: "Logged out",
      description: "See you again soon!",
    });
    navigate("/login");
  };

  // Get initials for avatar fallback
  const getInitials = (name: string) => {
    return name
      .split(' ')
      .map(part => part[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="max-w-2xl mx-auto space-y-6 animate-fade-in">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={handleFileChange}
      />

      {/* Profile Header */}
      <Card className="shadow-soft overflow-hidden">
        <div className="gradient-primary h-24" />
        <CardContent className="relative pb-6">
          <div className="absolute -top-12 left-6">
            <div className="relative">
              <Avatar className="w-24 h-24 border-4 border-card shadow-lg">
                <AvatarImage src={getProfilePictureUrl(profile.profilePictureUrl)} />
                <AvatarFallback className="text-2xl bg-primary text-primary-foreground">
                  {getInitials(profile.name) || "U"}
                </AvatarFallback>
              </Avatar>
              <button
                onClick={handleProfilePictureClick}
                disabled={isUploadingPicture}
                className="absolute bottom-0 right-0 p-2 bg-primary text-primary-foreground rounded-full shadow-lg hover:bg-primary/90 transition-colors disabled:opacity-50"
              >
                {isUploadingPicture ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Camera className="w-4 h-4" />
                )}
              </button>
            </div>
          </div>
          <div className="pt-14 flex items-start justify-between">
            <div>
              <h1 className="text-2xl font-bold">{profile.name || "User"}</h1>
              <p className="text-muted-foreground flex items-center gap-1">
                <MapPin className="w-4 h-4" />
                {profile.address || "No address set"}
              </p>
            </div>
            {!isEditing ? (
              <Button onClick={() => setIsEditing(true)} variant="outline" size="sm">
                <Edit2 className="w-4 h-4 mr-2" />
                {t('profile.editButton')}
              </Button>
            ) : (
              <Button onClick={handleSave} size="sm" disabled={isSaving}>
                {isSaving ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    {t('profile.savingButton')}
                  </>
                ) : (
                  <>
                    <Save className="w-4 h-4 mr-2" />
                    {t('profile.saveButton')}
                  </>
                )}
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Personal Information */}
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <User className="w-5 h-5 text-primary" />
            {t('profile.personalInfo.title')}
          </CardTitle>
          <CardDescription>{t('profile.personalInfo.description')}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid sm:grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="name">Full Name</Label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  id="name"
                  value={profile.name}
                  onChange={(e) => setProfile({ ...profile, name: e.target.value })}
                  disabled={!isEditing}
                  className="pl-9"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="email">Email Address</Label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  id="email"
                  type="email"
                  value={profile.email}
                  disabled={true}
                  className="pl-9 bg-muted"
                  title="Email cannot be changed"
                />
              </div>
            </div>
            <div className="space-y-2">
              <Label htmlFor="phone">Phone Number</Label>
              <div className="relative">
                <Phone className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  id="phone"
                  value={profile.phone}
                  onChange={(e) => setProfile({ ...profile, phone: e.target.value })}
                  disabled={!isEditing}
                  className="pl-9"
                />
              </div>
            </div>
            <div className="space-y-2 sm:col-span-2">
              <LocationSelector
                value={profile.address}
                onChange={(newAddress) => setProfile({ ...profile, address: newAddress })}
                disabled={!isEditing}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Preferences */}
      <Card className="shadow-soft">
        <CardHeader>
          <CardTitle className="text-lg flex items-center gap-2">
            <Languages className="w-5 h-5 text-primary" />
            {t('profile.preferences.title')}
          </CardTitle>
          <CardDescription>{t('profile.preferences.description')}</CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Languages className="w-5 h-5 text-muted-foreground" />
              <div>
                <p className="font-medium">{t('profile.preferences.language')}</p>
                <p className="text-sm text-muted-foreground">{t('profile.preferences.languageDesc')}</p>
              </div>
            </div>
            <Select value={i18n.language} onValueChange={(newLang) => {
              setProfile({ ...profile, language: newLang });
              i18n.changeLanguage(newLang);
              localStorage.setItem('language', newLang);
            }}>
              <SelectTrigger className="w-[140px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="en">English</SelectItem>
                <SelectItem value="hi">हिंदी</SelectItem>
                <SelectItem value="kn">ಕನ್ನಡ</SelectItem>
                <SelectItem value="te">తెలుగు</SelectItem>
                <SelectItem value="ta">தமிழ்</SelectItem>
                <SelectItem value="bn">বাংলা</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Bell className="w-5 h-5 text-muted-foreground" />
              <div>
                <p className="font-medium">{t('profile.preferences.notifications')}</p>
                <p className="text-sm text-muted-foreground">{t('profile.preferences.notificationsDesc')}</p>
              </div>
            </div>
            <Switch
              checked={profile.notificationsEnabled}
              onCheckedChange={handleNotificationToggle}
            />
          </div>

          <Separator />

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Moon className="w-5 h-5 text-muted-foreground" />
              <div>
                <p className="font-medium">{t('profile.preferences.darkMode')}</p>
                <p className="text-sm text-muted-foreground">{t('profile.preferences.darkModeDesc')}</p>
              </div>
            </div>
            <Switch checked={theme === 'dark'} onCheckedChange={toggleTheme} />
          </div>
        </CardContent>
      </Card>

      {/* Stats */}
      <Card className="bg-primary/5 border-primary/20">
        <CardContent className="p-6">
          <h3 className="font-semibold text-lg mb-4 flex items-center gap-2">
            <Leaf className="w-5 h-5 text-primary" />
            Your Statistics
          </h3>
          <div className="grid grid-cols-3 gap-4 text-center">
            <div className="p-4 bg-card rounded-lg shadow-soft">
              <p className="text-3xl font-bold text-primary">{stats.total}</p>
              <p className="text-sm text-muted-foreground">Total Scans</p>
            </div>
            <div className="p-4 bg-card rounded-lg shadow-soft">
              <p className="text-3xl font-bold text-success">{stats.healthy}</p>
              <p className="text-sm text-muted-foreground">Healthy</p>
            </div>
            <div className="p-4 bg-card rounded-lg shadow-soft">
              <p className="text-3xl font-bold text-destructive">{stats.diseased}</p>
              <p className="text-sm text-muted-foreground">Diseased</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Sign Out Button */}
      <Button
        onClick={handleLogout}
        variant="destructive"
        className="w-full"
      >
        <LogOut className="w-4 h-4 mr-2" />
        {t('profile.signOut')}
      </Button>
    </div>
  );
};
