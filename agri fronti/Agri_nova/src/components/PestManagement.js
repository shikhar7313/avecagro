import React, { useState, useEffect } from 'react';
import useText from '../hooks/useText';
import { Bug, AlertTriangle, Target, Map, Shield } from 'lucide-react';

const PestManagement = () => {
    const { t } = useText();
    // Mock data - you can replace with real API calls
    const [selectedCrop, setSelectedCrop] = useState('');
    const [severityLevel, setSeverityLevel] = useState('low');
    const [suggestedPests, setSuggestedPests] = useState([]);
    const [treatmentRecommendation, setTreatmentRecommendation] = useState('Select a crop and severity level to see recommendations.');

    // Crop data with associated pests
    const cropPestData = {
        'corn': ['Corn Borer', 'Armyworm', 'Cutworm', 'Aphids'],
        'wheat': ['Wheat Rust', 'Aphids', 'Hessian Fly', 'Wireworm'],
        'rice': ['Brown Planthopper', 'Rice Blast', 'Stem Borer', 'Leaf Folder'],
        'tomato': ['Whitefly', 'Hornworm', 'Aphids', 'Thrips'],
        'cotton': ['Bollworm', 'Whitefly', 'Aphids', 'Spider Mites'],
        'soybean': ['Soybean Aphid', 'Bean Leaf Beetle', 'Stink Bug', 'Spider Mites']
    };

    // Treatment recommendations based on pest and severity
    const treatmentData = {
        'corn': {
            'low': 'Monitor regularly and use pheromone traps. Apply neem oil if needed.',
            'medium': 'Use biological controls like Bt spray. Consider selective insecticides.',
            'high': 'Immediate intervention required. Use systemic insecticides and contact local extension service.'
        },
        'wheat': {
            'low': 'Regular field monitoring and crop rotation. Use resistant varieties.',
            'medium': 'Apply fungicides for rust. Use beneficial insects for aphid control.',
            'high': 'Emergency treatment with broad-spectrum fungicides and insecticides.'
        },
        'rice': {
            'low': 'Maintain proper water levels. Use light traps for monitoring.',
            'medium': 'Apply targeted insecticides. Use biocontrol agents.',
            'high': 'Systemic treatment required. Consider replanting severely affected areas.'
        },
        'tomato': {
            'low': 'Use yellow sticky traps. Apply organic sprays like soap solution.',
            'medium': 'Introduce beneficial insects. Use selective insecticides.',
            'high': 'Intensive treatment with systemic insecticides. Remove heavily infested plants.'
        },
        'cotton': {
            'low': 'Regular scouting and pheromone traps. Use Bt varieties.',
            'medium': 'Apply targeted bollworm sprays. Use beneficial predators.',
            'high': 'Emergency spraying program. Consider economic threshold levels.'
        },
        'soybean': {
            'low': 'Monitor with sweep nets. Use resistant varieties.',
            'medium': 'Apply insecticides at economic threshold. Use beneficial insects.',
            'high': 'Immediate treatment required. Use broad-spectrum insecticides.'
        }
    };

    // Handle crop selection
    const handleCropChange = (event) => {
        const crop = event.target.value;
        setSelectedCrop(crop);
        if (crop && cropPestData[crop]) {
            setSuggestedPests(cropPestData[crop]);
        } else {
            setSuggestedPests([]);
        }
        setTreatmentRecommendation('Select a crop and severity level to see recommendations.');
    };

    // Handle severity change
    const handleSeverityChange = (event) => {
        setSeverityLevel(event.target.value);
    };

    // Get treatment recommendation
    const getTreatmentRecommendation = () => {
        if (selectedCrop && treatmentData[selectedCrop] && treatmentData[selectedCrop][severityLevel]) {
            setTreatmentRecommendation(treatmentData[selectedCrop][severityLevel]);
        } else {
            setTreatmentRecommendation('Please select a valid crop and severity level.');
        }
    };

    return (
        <div className="p-6 space-y-6" data-scroll-section>
            {/* Header */}
            <div className="mb-6">
                <h1 className="text-3xl font-bold text-gray-100 hover:text-indigo-300 transition-colors duration-300 mb-2">Pest Management</h1>
                <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                    Experience next-level pest management with advanced features and stunning design.
                </p>
            </div>

            {/* Crop-Based Pest Suggestion and Pest Severity Input */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Crop-Based Pest Suggestion */}
                <div className="bg-white/10 backdrop-blur-md p-6 rounded-lg border border-white/20 hover:bg-white/15 transition-all duration-300">
                    <h2 className="text-xl font-bold mb-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300 flex items-center">
                        <Bug className="mr-3 text-green-400" size={24} />
                        Crop-Based Pest Suggestion
                    </h2>
                    <select
                        id="cropSelector"
                        value={selectedCrop}
                        onChange={handleCropChange}
                        className="w-full p-3 border border-white/20 rounded-lg mb-4 bg-white/10 text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        <option value="" className="bg-gray-800 text-white">Select a crop...</option>
                        <option value="corn" className="bg-gray-800 text-white">Corn</option>
                        <option value="wheat" className="bg-gray-800 text-white">Wheat</option>
                        <option value="rice" className="bg-gray-800 text-white">Rice</option>
                        <option value="tomato" className="bg-gray-800 text-white">Tomato</option>
                        <option value="cotton" className="bg-gray-800 text-white">Cotton</option>
                        <option value="soybean" className="bg-gray-800 text-white">Soybean</option>
                    </select>
                    <ul className="space-y-2">
                        {suggestedPests.length > 0 ? (
                            suggestedPests.map((pest, index) => (
                                <li key={index} className="flex items-start text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                                    <span className="text-red-400 mr-2 mt-1">•</span>
                                    {pest}
                                </li>
                            ))
                        ) : (
                            <li className="text-gray-400">No crop selected</li>
                        )}
                    </ul>
                </div>

                {/* Pest Severity Input */}
                <div className="bg-white/10 backdrop-blur-md p-6 rounded-lg border border-white/20 hover:bg-white/15 transition-all duration-300">
                    <h2 className="text-xl font-bold mb-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300 flex items-center">
                        <AlertTriangle className="mr-3 text-yellow-400" size={24} />
                        Pest Severity Level
                    </h2>
                    <select
                        id="severitySelector"
                        value={severityLevel}
                        onChange={handleSeverityChange}
                        className="w-full p-3 border border-white/20 rounded-lg mb-4 bg-white/10 text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                        <option value="low" className="bg-gray-800 text-white">Low</option>
                        <option value="medium" className="bg-gray-800 text-white">Medium</option>
                        <option value="high" className="bg-gray-800 text-white">High</option>
                    </select>
                    <button
                        onClick={getTreatmentRecommendation}
                        className="bg-red-600/80 hover:bg-red-600 text-white px-6 py-3 rounded-lg font-medium transition-all duration-300 w-full"
                    >
                        Get Recommendation
                    </button>
                </div>
            </div>

            {/* Live Infection Map */}
            <div className="bg-white/10 backdrop-blur-md p-6 rounded-lg border border-white/20 hover:bg-white/15 transition-all duration-300">
                <h2 className="text-xl font-bold mb-4 text-gray-100 hover:text-indigo-300 transition-colors duration-300 flex items-center">
                    <Map className="mr-3 text-purple-400" size={24} />
                    Live Infection Map
                </h2>
                <div className="bg-white/5 rounded-lg p-8 text-center h-64 flex items-center justify-center border border-white/10">
                    <div className="text-center">
                        <Shield className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                        <p className="text-gray-100 hover:text-indigo-300 transition-colors duration-300 mb-2">
                            Interactive Infection Map
                        </p>
                        <p className="text-gray-400 text-sm">
                            Real-time pest infection data visualization will be displayed here
                        </p>
                    </div>
                </div>
            </div>

            {/* Additional Information Cards */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Prevention Tips */}
                <div className="bg-white/10 backdrop-blur-md p-6 rounded-lg border border-white/20 hover:bg-white/15 transition-all duration-300">
                    <h3 className="text-lg font-bold mb-3 text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                        Prevention Tips
                    </h3>
                    <ul className="space-y-2 text-sm">
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Regular field monitoring</li>
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Crop rotation practices</li>
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Use resistant varieties</li>
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Maintain field hygiene</li>
                    </ul>
                </div>

                {/* Early Warning Signs */}
                <div className="bg-white/10 backdrop-blur-md p-6 rounded-lg border border-white/20 hover:bg-white/15 transition-all duration-300">
                    <h3 className="text-lg font-bold mb-3 text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                        Early Warning Signs
                    </h3>
                    <ul className="space-y-2 text-sm">
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Leaf discoloration</li>
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Unusual insect activity</li>
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Plant wilting</li>
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Growth abnormalities</li>
                    </ul>
                </div>

                {/* Treatment Recommendations */}
                <div className="bg-white/10 backdrop-blur-md p-6 rounded-lg border border-white/20 hover:bg-white/15 transition-all duration-300">
                    <h3 className="text-lg font-bold mb-3 text-gray-100 hover:text-indigo-300 transition-colors duration-300">
                        Treatment Recommendations
                    </h3>
                    <ul className="space-y-2 text-sm">
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Apply organic pesticides first</li>
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Use biological control agents</li>
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Targeted chemical spraying</li>
                        <li className="text-gray-100 hover:text-indigo-300 transition-colors duration-300">• Integrated pest management</li>
                    </ul>
                </div>
            </div>
        </div>
    );
};

export default PestManagement;