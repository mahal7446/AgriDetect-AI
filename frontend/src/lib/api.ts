/**
 * API configuration and client functions
 */

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// Debug: Log the API URL being used
console.log('🔗 API Base URL:', API_BASE_URL);
console.log('🔗 Environment VITE_API_URL:', import.meta.env.VITE_API_URL);

export interface PredictionResponse {
  success: boolean;
  diseaseName: string;
  confidence: number;
  cropName: string;
  severity: "Low" | "Medium" | "High";
  allPredictions?: Array<{
    disease: string;
    confidence: number;
  }>;
  description: string;
  error?: string;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

/**
 * Check if the backend API is healthy and model is loaded
 */
export const checkHealth = async (): Promise<HealthResponse> => {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error('Health check failed');
  }
  return response.json();
};

/**
 * Predict plant disease from an image file
 */
export const predictDisease = async (imageFile: File): Promise<PredictionResponse> => {
  const formData = new FormData();
  formData.append('image', imageFile);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
    throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
  }

  return response.json();
};

// ============== PROFILE API ==============

export interface ProfileData {
  email: string;
  fullName: string;
  phone: string;
  address?: string;
  profilePictureUrl?: string;
}

export interface ApiResponse<T = any> {
  success: boolean;
  message?: string;
  error?: string;
  user?: T;
}

/**
 * Fetch user profile data
 */
export const getProfile = async (email: string): Promise<ApiResponse<ProfileData>> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/profile/get?email=${encodeURIComponent(email)}`);
    const data = await response.json();
    return data;
  } catch (error) {
    return {
      success: false,
      error: 'Failed to fetch profile data'
    };
  }
};

/**
 * Update user profile information
 */
export const updateProfile = async (email: string, profileData: Partial<ProfileData>): Promise<ApiResponse<ProfileData>> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/profile/update`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email,
        ...profileData
      }),
    });
    const data = await response.json();
    return data;
  } catch (error) {
    return {
      success: false,
      error: 'Failed to update profile'
    };
  }
};

/**
 * Upload profile picture
 */
export const uploadProfilePicture = async (email: string, file: File): Promise<ApiResponse> => {
  try {
    const formData = new FormData();
    formData.append('email', email);
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/api/profile/upload-picture`, {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    return data;
  } catch (error) {
    return {
      success: false,
      error: 'Failed to upload profile picture'
    };
  }
};

/**
 * Get full profile picture URL
 */
export const getProfilePictureUrl = (profilePictureUrl?: string | null): string => {
  if (!profilePictureUrl) return '';
  if (profilePictureUrl.startsWith('http')) return profilePictureUrl;
  return `${API_BASE_URL}${profilePictureUrl}`;
};

/**
 * Fetch user scan statistics
 */
export const getUserStats = async (email: string): Promise<{ success: boolean; stats: { total: number; healthy: number; diseased: number }; error?: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/profile/stats?email=${encodeURIComponent(email)}`);
    const data = await response.json();
    return data;
  } catch (error) {
    return {
      success: false,
      stats: { total: 0, healthy: 0, diseased: 0 },
      error: 'Failed to fetch statistics'
    };
  }
};

// ============== HISTORY API ==============

export interface HistoryItem {
  id: number;
  diseaseName: string;
  confidence: number;
  cropName: string;
  severity: "Low" | "Medium" | "High";
  imageUrl: string | null;
  scanDate: string;
}

export interface HistoryResponse {
  success: boolean;
  history: HistoryItem[];
  count: number;
  error?: string;
}

/**
 * Save scan to user's history
 */
export const saveScanToHistory = async (
  email: string,
  prediction: PredictionResponse,
  imageFile: File
): Promise<ApiResponse> => {
  try {
    const formData = new FormData();
    formData.append('email', email);
    formData.append('diseaseName', prediction.diseaseName);
    formData.append('confidence', prediction.confidence.toString());
    formData.append('cropName', prediction.cropName);
    formData.append('severity', prediction.severity);
    formData.append('image', imageFile);

    const response = await fetch(`${API_BASE_URL}/api/history/save`, {
      method: 'POST',
      body: formData,
    });

    return response.json();
  } catch (error) {
    return {
      success: false,
      error: 'Failed to save scan to history'
    };
  }
};

/**
 * Get user's scan history
 */
export const getHistory = async (email: string, limit: number = 50): Promise<HistoryResponse> => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/history/get?email=${encodeURIComponent(email)}&limit=${limit}`
    );
    return response.json();
  } catch (error) {
    return {
      success: false,
      history: [],
      count: 0,
      error: 'Failed to fetch history'
    };
  }
};

/**
 * Delete a scan from history
 */
export const deleteHistoryItem = async (scanId: number, email: string): Promise<ApiResponse> => {
  try {
    const response = await fetch(
      `${API_BASE_URL}/api/history/delete/${scanId}?email=${encodeURIComponent(email)}`,
      { method: 'DELETE' }
    );
    return response.json();
  } catch (error) {
    return {
      success: false,
      error: 'Failed to delete scan'
    };
  }
};

/**
 * Get full scan image URL
 */
export const getScanImageUrl = (imageUrl?: string | null): string => {
  if (!imageUrl) return '/placeholder.svg';
  if (imageUrl.startsWith('http')) return imageUrl;
  return `${API_BASE_URL}${imageUrl}`;
};

// ============== COMMUNITY ALERTS API ==============

export interface CommunityAlert {
  id: number;
  farmerName: string;
  location: string;
  diseaseReported: string;
  description?: string;
  preventionMethods?: string;
  imageUrl?: string;
  userEmail?: string;
  createdAt: string;
}

/**
 * Get recent alerts
 */
export const getRecentAlerts = async (limit: number = 10): Promise<{ success: boolean; alerts: CommunityAlert[]; error?: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/alerts/recent?limit=${limit}`);
    return response.json();
  } catch (error) {
    return { success: false, alerts: [], error: 'Failed to fetch alerts' };
  }
};

/**
 * Get alerts by location
 */
export const getAlertsByLocation = async (email: string, limit: number = 20): Promise<{ success: boolean; alerts: CommunityAlert[]; error?: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/alerts/by-location?email=${encodeURIComponent(email)}&limit=${limit}`);
    return response.json();
  } catch (error) {
    return { success: false, alerts: [], error: 'Failed to fetch alerts by location' };
  }
};

/**
 * Submit a new community alert
 */
export const submitCommunityAlert = async (formData: FormData): Promise<ApiResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/alerts/submit`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  } catch (error) {
    return { success: false, error: 'Failed to submit alert' };
  }
};

/**
 * Update a community alert
 */
export const updateCommunityAlert = async (alertId: number, formData: FormData): Promise<ApiResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/alerts/update/${alertId}`, {
      method: 'POST',
      body: formData,
    });
    return response.json();
  } catch (error) {
    return { success: false, error: 'Failed to update alert' };
  }
};

/**
 * Delete a community alert
 */
export const deleteCommunityAlert = async (alertId: number, email: string): Promise<ApiResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/alerts/delete/${alertId}?email=${encodeURIComponent(email)}`, {
      method: 'DELETE',
    });
    return response.json();
  } catch (error) {
    return { success: false, error: 'Failed to delete alert' };
  }
};

/**
 * Get new alerts count for polling
 */
export const getNewAlertsCount = async (email: string, lastSeenId: number): Promise<{ success: boolean; count: number; error?: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/alerts/new-count?email=${encodeURIComponent(email)}&lastSeenId=${lastSeenId}`);
    return response.json();
  } catch (error) {
    return { success: false, count: 0, error: 'Failed to fetch new alerts count' };
  }
};

/**
 * Get notification preference
 */
export const getNotificationPreference = async (email: string): Promise<{ success: boolean; enabled: boolean; error?: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/profile/notification-preference?email=${encodeURIComponent(email)}`);
    return response.json();
  } catch (error) {
    return { success: false, enabled: true, error: 'Failed to fetch notification preference' };
  }
};

/**
 * Update notification preference
 */
export const updateNotificationPreference = async (email: string, enabled: boolean): Promise<ApiResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/profile/update-notification-preference`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, enabled }),
    });
    return response.json();
  } catch (error) {
    return { success: false, error: 'Failed to update notification preference' };
  }
};
// ============== ANALYTICS API ==============

export interface AnalyticsSummary {
  totalScans: number;
  averageHealth: number;
  diseaseAlerts: number;
  estimatedYield: number;
}

export interface AnalyticsCharts {
  yieldForecast: Array<{ crop: string; yield: number }>;
  diseaseTrends: Array<{ month: string; healthy: number; diseased: number }>;
}

export interface AnalyticsReport {
  id: number;
  date: string;
  cropType: string;
  diagnosis: string;
  riskLevel: string;
}

/**
 * Get analytics summary
 */
export const getAnalyticsSummary = async (email: string): Promise<{ success: boolean; summary: AnalyticsSummary; error?: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analytics/summary?email=${encodeURIComponent(email)}`);
    return response.json();
  } catch (error) {
    return { success: false, summary: { totalScans: 0, averageHealth: 0, diseaseAlerts: 0, estimatedYield: 0 }, error: 'Failed to fetch analytics summary' };
  }
};

/**
 * Get analytics charts data
 */
export const getAnalyticsCharts = async (email: string): Promise<{ success: boolean; charts: AnalyticsCharts; error?: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analytics/charts?email=${encodeURIComponent(email)}`);
    return response.json();
  } catch (error) {
    return { success: false, charts: { yieldForecast: [], diseaseTrends: [] }, error: 'Failed to fetch analytics charts' };
  }
};

/**
 * Get analytics reports
 */
export const getAnalyticsReports = async (email: string, limit: number = 10): Promise<{ success: boolean; reports: AnalyticsReport[]; error?: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/analytics/reports?email=${encodeURIComponent(email)}&limit=${limit}`);
    return response.json();
  } catch (error) {
    return { success: false, reports: [], error: 'Failed to fetch analytics reports' };
  }
};
