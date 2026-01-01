import React, { useState } from 'react';
import { X } from 'lucide-react';

const NewUserForm = ({ isOpen, onClose, onSubmit }) => {
    const [formData, setFormData] = useState({
        primaryCrop: '',
        lastCropPlanted: '',
        farmSize: '',
        soilType: '',
        soilTestReport: '',
        soilReportFile: null,
        irrigationAccess: '',
        farmingMethod: '',
        nextSeason: '',
        cropResidueHandling: '',
        sunlightCondition: '',
        intercroppingOpen: ''
    });

    const [showFileUpload, setShowFileUpload] = useState(false);

    const handleInputChange = (e) => {
        const { id, value, type, files } = e.target;

        if (type === 'file') {
            setFormData(prev => ({
                ...prev,
                [id]: files[0]
            }));
        } else {
            setFormData(prev => ({
                ...prev,
                [id]: value
            }));
        }

        // Show/hide file upload based on soil test report selection
        if (id === 'soilTestReport') {
            setShowFileUpload(value === 'Yes');
            if (value === 'No') {
                setFormData(prev => ({
                    ...prev,
                    soilReportFile: null
                }));
            }
        }
    };

    const handleSubmit = (e) => {
        e.preventDefault();

        // Create user data object
        const userData = {
            id: Date.now(),
            dateCreated: new Date().toISOString(),
            isNewUser: false, // Mark as existing user after form submission
            ...formData,
            soilReportFile: formData.soilReportFile ? formData.soilReportFile.name : null
        };

        onSubmit(userData);
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
            <div className="bg-white p-6 rounded-lg shadow-lg w-96 max-h-[90vh] overflow-y-auto relative">
                {/* Close button */}
                <button
                    onClick={onClose}
                    className="absolute top-4 right-4 text-gray-500 hover:text-gray-700"
                >
                    <X className="w-6 h-6" />
                </button>

                <h2 className="text-2xl font-bold mb-4 text-center text-gray-800">
                    Welcome! Tell us about yourself
                </h2>

                <form onSubmit={handleSubmit} className="space-y-4">
                    {/* Primary Crop */}
                    <label className="block">
                        <span className="text-gray-700 font-semibold">What is your primary crop currently?</span>
                        <input
                            type="text"
                            id="primaryCrop"
                            value={formData.primaryCrop}
                            onChange={handleInputChange}
                            className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            placeholder="e.g., Corn, Wheat"
                            required
                        />
                    </label>

                    {/* Last Season Crop */}
                    <label className="block">
                        <span className="text-gray-700 font-semibold">What crop did you grow last season?</span>
                        <input
                            type="text"
                            id="lastCropPlanted"
                            value={formData.lastCropPlanted}
                            onChange={handleInputChange}
                            className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            placeholder="e.g., Rice"
                            required
                        />
                    </label>

                    {/* Farm Size */}
                    <label className="block">
                        <span className="text-gray-700 font-semibold">How many acres of land do you farm?</span>
                        <input
                            type="number"
                            id="farmSize"
                            value={formData.farmSize}
                            onChange={handleInputChange}
                            className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            placeholder="e.g., 50"
                            required
                        />
                    </label>

                    {/* Soil Type */}
                    <label className="block">
                        <span className="text-gray-700 font-semibold">What is your soil type?</span>
                        <input
                            type="text"
                            id="soilType"
                            value={formData.soilType}
                            onChange={handleInputChange}
                            className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            placeholder="e.g., Loamy, Sandy, Clayey"
                            required
                        />
                    </label>

                    {/* Soil Testing Report */}
                    <label className="block">
                        <span className="text-gray-700 font-semibold">Do you have a recent soil testing report?</span>
                        <select
                            id="soilTestReport"
                            value={formData.soilTestReport}
                            onChange={handleInputChange}
                            className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            required
                        >
                            <option value="" disabled>Select an option</option>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                        </select>
                    </label>

                    {/* Conditional File Upload */}
                    {showFileUpload && (
                        <label className="block">
                            <span className="text-gray-700 font-semibold">Upload your soil testing report (PDF, JPG, PNG):</span>
                            <input
                                type="file"
                                id="soilReportFile"
                                onChange={handleInputChange}
                                accept=".pdf,.jpg,.jpeg,.png"
                                className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            />
                        </label>
                    )}

                    {/* Irrigation Access */}
                    <label className="block">
                        <span className="text-gray-700 font-semibold">Describe your irrigation access:</span>
                        <input
                            type="text"
                            id="irrigationAccess"
                            value={formData.irrigationAccess}
                            onChange={handleInputChange}
                            className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            placeholder="e.g., Drip, Flood, None"
                            required
                        />
                    </label>

                    {/* Farming Method */}
                    <label className="block">
                        <span className="text-gray-700 font-semibold">What is your preferred farming method?</span>
                        <select
                            id="farmingMethod"
                            value={formData.farmingMethod}
                            onChange={handleInputChange}
                            className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            required
                        >
                            <option value="" disabled>Select a method</option>
                            <option value="Organic">Organic</option>
                            <option value="Conventional">Conventional</option>
                            <option value="Integrated">Integrated Pest Management</option>
                        </select>
                    </label>

                    {/* Next Season */}
                    <label className="block">
                        <span className="text-gray-700 font-semibold">Which season are you planning for next?</span>
                        <input
                            type="text"
                            id="nextSeason"
                            value={formData.nextSeason}
                            onChange={handleInputChange}
                            className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            placeholder="e.g., Kharif, Rabi, Zaid"
                            required
                        />
                    </label>

                    {/* Crop Residue Handling */}
                    <label className="block">
                        <span className="text-gray-700 font-semibold">Was crop residue left in the field or burned?</span>
                        <select
                            id="cropResidueHandling"
                            value={formData.cropResidueHandling}
                            onChange={handleInputChange}
                            className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            required
                        >
                            <option value="" disabled>Select an option</option>
                            <option value="Left">Left</option>
                            <option value="Burned">Burned</option>
                        </select>
                    </label>

                    {/* Sunlight Condition */}
                    <label className="block">
                        <span className="text-gray-700 font-semibold">Describe the sunlight condition on your farm:</span>
                        <input
                            type="text"
                            id="sunlightCondition"
                            value={formData.sunlightCondition}
                            onChange={handleInputChange}
                            className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            placeholder="e.g., Full Sun, Partial Shade, Mostly Shade"
                            required
                        />
                    </label>

                    {/* Intercropping */}
                    <label className="block">
                        <span className="text-gray-700 font-semibold">Are you open to intercropping or multi-layer farming techniques?</span>
                        <select
                            id="intercroppingOpen"
                            value={formData.intercroppingOpen}
                            onChange={handleInputChange}
                            className="w-full mt-1 p-3 border rounded-lg shadow-sm focus:ring-2 focus:ring-green-500 focus:outline-none text-black"
                            required
                        >
                            <option value="" disabled>Select an option</option>
                            <option value="Yes">Yes</option>
                            <option value="No">No</option>
                            <option value="Maybe">Maybe</option>
                        </select>
                    </label>

                    <button
                        type="submit"
                        className="w-full bg-green-600 text-white py-3 rounded-lg font-semibold shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 transition-colors"
                    >
                        Submit
                    </button>
                </form>
            </div>
        </div>
    );
};

export default NewUserForm;