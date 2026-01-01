// Utility functions for user data management

const QUESTIONNAIRE_FARM_SIZE_KEY = 'questionnaireFarmSize'

// Load initial userData from JSON file
export const loadInitialUserData = async () => {
    try {
        const response = await fetch('/src/data/dashboard/userData.json');
        if (!response.ok) {
            return [];
        }
        const userData = await response.json();
        return Array.isArray(userData) ? userData.filter(user => user !== null) : [];
    } catch (error) {
        console.error('Error loading initial user data:', error);
        return [];
    }
};

// Check if user exists in userData
export const checkUserExists = async (username) => {
    try {
        // Check localStorage first
        const localData = localStorage.getItem('userData');
        let users = [];

        if (localData) {
            users = JSON.parse(localData);
        } else {
            // Load from JSON file if localStorage is empty
            users = await loadInitialUserData();
            localStorage.setItem('userData', JSON.stringify(users));
        }

        return users.some(user => user && user.username === username);
    } catch (error) {
        console.error('Error checking user existence:', error);
        return false;
    }
};

// Save new user data
export const saveUserData = async (userData) => {
    try {
        // Get existing data from localStorage or load from JSON
        let users = [];
        const existingData = localStorage.getItem('userData');

        if (existingData) {
            users = JSON.parse(existingData);
        } else {
            users = await loadInitialUserData();
        }

        // Add new user data with username
        const newUserData = {
            username: localStorage.getItem('username') || 'user',
            ...userData
        };

        users.push(newUserData);

        // Save to localStorage
        localStorage.setItem('userData', JSON.stringify(users));

        // Also mark user as not new in separate flag
        localStorage.setItem(`user_${localStorage.getItem('username')}_isNew`, 'false');

        return true;
    } catch (error) {
        console.error('Error saving user data:', error);
        return false;
    }
};

// Check if current user is new
export const isNewUser = (username) => {
    // Check if user has completed the form
    const hasCompletedForm = localStorage.getItem(`user_${username}_isNew`);
    return hasCompletedForm !== 'false';
};

// Get user data
export const getUserData = (username) => {
    try {
        const data = localStorage.getItem('userData');
        if (!data) return null;

        const users = JSON.parse(data);
        return users.find(user => user && user.username === username) || null;
    } catch (error) {
        console.error('Error getting user data:', error);
        return null;
    }
};

// Get current user's farm size in acres
export const getCurrentUserFarmSize = () => {
    try {
        const storedQuestionnaireSize = localStorage.getItem(QUESTIONNAIRE_FARM_SIZE_KEY);
        if (storedQuestionnaireSize) {
            const parsed = JSON.parse(storedQuestionnaireSize);
            const value = typeof parsed === 'object' ? parsed.value : parsed;
            const num = parseFloat(value);
            if (!isNaN(num)) {
                return num;
            }
        }

        const username = localStorage.getItem('username') || 'user';
        const userData = getUserData(username);

        if (!userData || !userData.farmSize) {
            return 0; // Default to 0 if no data
        }

        // Parse the farm size (assuming it's stored as a number or string)
        const farmSize = parseFloat(userData.farmSize);
        return isNaN(farmSize) ? 0 : farmSize;
    } catch (error) {
        console.error('Error getting farm size:', error);
        return 0;
    }
};

// Check if user should see IoT features
export const shouldShowIoTFeatures = () => {
    const farmSize = getCurrentUserFarmSize();
    return farmSize >= 2; // Show IoT only if farm is 2+ acres
};

// Check if user should see drone features
export const shouldShowDroneFeatures = () => {
    const farmSize = getCurrentUserFarmSize();
    return farmSize >= 7; // Show drone only if farm is 7+ acres
};

// Get farm size category
export const getFarmSizeCategory = () => {
    const farmSize = getCurrentUserFarmSize();

    if (farmSize < 2) {
        return 'small'; // Small farm - no IoT
    } else if (farmSize < 7) {
        return 'medium'; // Medium farm - IoT but no drone
    } else {
        return 'large'; // Large farm - full IoT including drone
    }
};

export const storeQuestionnaireFarmSize = (acres) => {
    try {
        const payload = {
            value: acres,
            updatedAt: new Date().toISOString()
        };
        localStorage.setItem(QUESTIONNAIRE_FARM_SIZE_KEY, JSON.stringify(payload));
    } catch (error) {
        console.error('Error storing questionnaire farm size:', error);
    }
};