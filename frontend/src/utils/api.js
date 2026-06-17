export let API_BASE_URL = localStorage.getItem("visionsafe_api_url") || "http://127.0.0.1:8000";

/**
 * Updates the API Base URL both in memory and local storage.
 * @param {string} url - New API Base URL
 */
export const setApiUrl = (url) => {
    API_BASE_URL = url;
    localStorage.setItem("visionsafe_api_url", url);
};

export const getApiUrl = () => API_BASE_URL;

/**
 * Detects activity/objects in a file or frame.
 * @param {File} file - Image or Video file
 * @param {string} mode - 'file' (upload) or 'frame' (live/playback)
 * @returns {Promise<Object>}
 */
export const detectMedia = async (file, mode = 'frame') => {
    const formData = new FormData();

    // API expects 'file' for uploads, 'frame' for single frame inference
    const key = mode === 'file' ? 'file' : 'frame';
    formData.append(key, file);

    const endpoint = mode === 'file' ? '/detect' : '/detect_frame';

    try {
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            throw new Error(`Server Error: ${response.status}`);
        }

        return await response.json();
    } catch (error) {
        console.error("API Call Failed:", error);
        throw error;
    }
};

export const checkHealth = async () => {
    try {
        const res = await fetch(`${API_BASE_URL}/health`);
        return await res.json();
    } catch (e) {
        return { status: "error", details: e };
    }
}
