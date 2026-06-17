export const addToHistory = (record) => {
    const existing = JSON.parse(localStorage.getItem("visionsafe_history") || "[]");
    const newRecord = {
        ...record,
        id: Date.now(),
        timestamp: new Date().toISOString()
    };
    // Keep max 50 records
    const updated = [newRecord, ...existing].slice(0, 50);
    localStorage.setItem("visionsafe_history", JSON.stringify(updated));
};

export const getHistory = () => {
    return JSON.parse(localStorage.getItem("visionsafe_history") || "[]");
};

export const clearHistory = () => {
    localStorage.removeItem("visionsafe_history");
};
