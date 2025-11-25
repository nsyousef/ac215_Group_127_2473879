import { BASE_API_URL, uuid } from "./Common";
import axios from 'axios';

console.log("BASE_API_URL:", BASE_API_URL)

// Create an axios instance with base configuration
const api = axios.create({
    baseURL: BASE_API_URL
});

const DataService = {
    Init: function () {
        // Any application initialization logic comes here
    },
    MockAPICall: async function (formData) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1000));

        // Mock response data
        const mockResults = {
            success: true
        };

        return Promise.resolve({ data: mockResults });
    }
}

export default DataService;
