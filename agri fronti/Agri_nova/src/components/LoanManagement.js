import React, { useState, useEffect } from 'react';

const LoanManagement = () => {
    const [schemes, setSchemes] = useState([]);
    const [filteredSchemes, setFilteredSchemes] = useState([]);
    const [selectedSchemeType, setSelectedSchemeType] = useState('all');
    const [selectedState, setSelectedState] = useState('');
    const [availableStates, setAvailableStates] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Fetch loan schemes from API
    const fetchLoanSchemes = async () => {
        setLoading(true);
        setError(null);

        try {
            // Use the agri_schemes_india.json as the source
            const apiUrl = '/datasets/agri_schemes_india.json';
            const res = await fetch(apiUrl);

            if (!res.ok) {
                throw new Error(`API error: ${res.status}`);
            }

            const schemesData = await res.json();

            if (!Array.isArray(schemesData) || schemesData.length === 0) {
                setError('No schemes found.');
                setSchemes([]);
                setFilteredSchemes([]);
                return;
            }

            setSchemes(schemesData);
            setFilteredSchemes(schemesData);

            // Extract unique states for dropdown
            const states = Array.from(new Set(
                schemesData
                    .filter(s => s.category === 'State' && s.state)
                    .map(s => s.state)
            )).sort();
            setAvailableStates(states);

        } catch (err) {
            console.error('Failed to load schemes:', err);
            setError(`Failed to load schemes (Network error: ${err.message})`);
            // Fallback to local data if API fails
            try {
                const agriSchemesData = await import('../data/dashboard/agri_schemes_india.json');
                setSchemes(agriSchemesData.default);
                setFilteredSchemes(agriSchemesData.default);

                const states = Array.from(new Set(
                    agriSchemesData.default
                        .filter(s => s.category === 'State' && s.state)
                        .map(s => s.state)
                )).sort();
                setAvailableStates(states);
                setError(null);
            } catch (fallbackErr) {
                console.error('Failed to load fallback data:', fallbackErr);
            }
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchLoanSchemes();
    }, []);

    // Filter schemes based on type and state
    const filterSchemes = (schemeType = selectedSchemeType, state = selectedState) => {
        let filtered = schemes;

        if (schemeType === 'National') {
            filtered = schemes.filter(s => s.category === 'National');
        } else if (schemeType === 'State') {
            filtered = schemes.filter(s => s.category === 'State' && (!state || s.state === state));
        }
        // 'all' shows everything, so no additional filtering needed

        setFilteredSchemes(filtered);
    };

    const handleSchemeTypeChange = (event) => {
        const schemeType = event.target.value;
        setSelectedSchemeType(schemeType);
        setSelectedState(''); // Reset state selection when scheme type changes
        filterSchemes(schemeType, '');
    };

    const handleStateChange = (event) => {
        const state = event.target.value;
        setSelectedState(state);
        filterSchemes(selectedSchemeType, state);
    };

    // Enhanced scheme card renderer
    const renderSchemeCard = (scheme) => {
        const formatBenefit = (benefit) => {
            if (typeof benefit === 'string') return benefit;
            if (typeof benefit === 'object') {
                return Object.entries(benefit).map(([key, value]) =>
                    `${key}: ${value}`
                ).join(', ');
            }
            return 'Contact local office for details';
        };

        return (
            <div key={scheme.scheme_name} className="bg-white/10 backdrop-blur-md rounded-lg p-6 border border-white/20 hover:bg-white/15 transition-all duration-300 mb-4">
                <div className="mb-4">
                    <h4 className="text-xl font-semibold text-white mb-2">{scheme.scheme_name || ''}</h4>
                    <div className="flex gap-2 mb-3">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium ${scheme.category === 'National'
                                ? 'bg-blue-500/30 text-blue-200'
                                : 'bg-green-500/30 text-green-200'
                            }`}>
                            {scheme.category}
                        </span>
                        {scheme.state && (
                            <span className="px-3 py-1 rounded-full text-sm font-medium bg-purple-500/30 text-purple-200">
                                {scheme.state}
                            </span>
                        )}
                    </div>
                </div>

                <div className="space-y-3">
                    <div>
                        <h5 className="text-sm font-semibold text-gray-300 mb-1">Purpose:</h5>
                        <p className="text-white text-sm">{scheme.purpose || ''}</p>
                    </div>

                    {scheme.loan_amount_range && (
                        <div>
                            <h5 className="text-sm font-semibold text-gray-300 mb-1">Loan Amount Range:</h5>
                            <p className="text-white text-sm">{scheme.loan_amount_range}</p>
                        </div>
                    )}

                    {scheme.interest_rate && (
                        <div>
                            <h5 className="text-sm font-semibold text-gray-300 mb-1">Interest Rate:</h5>
                            <p className="text-white text-sm">{scheme.interest_rate}</p>
                        </div>
                    )}

                    {scheme.benefit && (
                        <div>
                            <h5 className="text-sm font-semibold text-gray-300 mb-1">Benefit:</h5>
                            <p className="text-white text-sm">{formatBenefit(scheme.benefit)}</p>
                        </div>
                    )}

                    <div>
                        <h5 className="text-sm font-semibold text-gray-300 mb-1">Eligibility:</h5>
                        <p className="text-white text-sm">{scheme.eligibility || ''}</p>
                    </div>

                    {scheme.application_process && (
                        <div>
                            <h5 className="text-sm font-semibold text-gray-300 mb-2">Application Process:</h5>
                            <div className="bg-white/5 rounded-md p-3 max-h-40 overflow-y-auto">
                                {scheme.application_process.steps ? (
                                    <ol className="list-decimal list-inside space-y-1 text-xs text-gray-200">
                                        {scheme.application_process.steps.map((step, index) => (
                                            <li key={index}>{step}</li>
                                        ))}
                                    </ol>
                                ) : (
                                    <p className="text-xs text-gray-200">Contact local agriculture office for application process</p>
                                )}
                            </div>

                            <div className="flex flex-wrap gap-2 mt-2">
                                {scheme.application_process.online_link && (
                                    <a
                                        href={scheme.application_process.online_link}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="bg-blue-600/40 hover:bg-blue-600/60 text-blue-200 px-3 py-1 rounded-md text-xs font-medium transition-all duration-200"
                                    >
                                        Apply / More Info
                                    </a>
                                )}
                                {scheme.application_process.summary_portal && (
                                    <a
                                        href={scheme.application_process.summary_portal}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className="bg-purple-600/40 hover:bg-purple-600/60 text-purple-200 px-3 py-1 rounded-md text-xs font-medium transition-all duration-200"
                                    >
                                        Summary Portal
                                    </a>
                                )}
                            </div>
                        </div>
                    )}

                    {scheme.help_links && scheme.help_links.length > 0 && (
                        <div>
                            <h5 className="text-sm font-semibold text-gray-300 mb-2">Help Links:</h5>
                            <ul className="space-y-1">
                                {scheme.help_links.map((link, index) => (
                                    <li key={index} className="flex items-center gap-2">
                                        <a
                                            href={link.url}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className="text-green-400 hover:text-green-300 text-xs underline"
                                        >
                                            {link.title || link.type || 'Help Link'}
                                        </a>
                                        {link.type && (
                                            <span className="text-gray-500 text-xs">({link.type})</span>
                                        )}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            </div>
        );
    };

    // Group schemes by category and state
    const renderGroupedSchemes = () => {
        if (filteredSchemes.length === 0) {
            return (
                <div className="col-span-full text-center py-12">
                    <div className="bg-white/5 backdrop-blur-md rounded-lg p-8 border border-white/10">
                        <p className="text-red-400 text-lg">No schemes found for the selected filter.</p>
                        <p className="text-gray-500 text-sm mt-2">Try selecting different filters or contact your local agriculture office.</p>
                    </div>
                </div>
            );
        }

        const content = [];

        // National Schemes
        if (selectedSchemeType === 'all' || selectedSchemeType === 'National') {
            const nationalSchemes = filteredSchemes.filter(s => s.category === 'National');
            if (nationalSchemes.length > 0) {
                content.push(
                    <div key="national-section" className="col-span-full mb-6">
                        <h2 className="text-2xl font-bold mb-4 text-green-400">National Schemes</h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {nationalSchemes.map(scheme => renderSchemeCard(scheme))}
                        </div>
                    </div>
                );
            }
        }

        // State Schemes
        if (selectedSchemeType === 'all' || selectedSchemeType === 'State') {
            const stateSchemes = {};
            filteredSchemes.filter(s => s.category === 'State').forEach(s => {
                if (!stateSchemes[s.state]) stateSchemes[s.state] = [];
                stateSchemes[s.state].push(s);
            });

            if (Object.keys(stateSchemes).length > 0) {
                content.push(
                    <div key="state-section" className="col-span-full">
                        <h2 className="text-2xl font-bold mt-8 mb-4 text-blue-400">State Schemes</h2>
                        {Object.keys(stateSchemes).sort().map(state => (
                            <div key={state} className="mb-8">
                                <h3 className="text-xl font-semibold mb-4 text-blue-300">{state}</h3>
                                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                                    {stateSchemes[state].map(scheme => renderSchemeCard(scheme))}
                                </div>
                            </div>
                        ))}
                    </div>
                );
            }
        }

        return content;
    };

    return (
        <div className="p-6 space-y-6" data-scroll-section>
            <div className="bg-white/5 backdrop-blur-md rounded-lg p-6 border border-white/10">
                <h1 className="text-3xl font-bold text-white mb-4">Loan Management</h1>
                <p className="text-gray-300 mb-6">
                    Select a scheme type and view details for farm loans and subsidies.
                </p>

                {/* Filter Controls */}
                <div className="mb-6 flex flex-wrap gap-4 items-center">
                    <div className="flex flex-wrap gap-4">
                        <label className="flex items-center text-white cursor-pointer">
                            <input
                                type="radio"
                                name="schemeType"
                                value="all"
                                checked={selectedSchemeType === 'all'}
                                onChange={handleSchemeTypeChange}
                                className="mr-2 text-blue-500 focus:ring-blue-500"
                            />
                            <span className="font-semibold">All Schemes</span>
                        </label>

                        <label className="flex items-center text-white cursor-pointer">
                            <input
                                type="radio"
                                name="schemeType"
                                value="National"
                                checked={selectedSchemeType === 'National'}
                                onChange={handleSchemeTypeChange}
                                className="mr-2 text-blue-500 focus:ring-blue-500"
                            />
                            <span className="font-semibold">National Schemes</span>
                        </label>

                        <label className="flex items-center text-white cursor-pointer">
                            <input
                                type="radio"
                                name="schemeType"
                                value="State"
                                checked={selectedSchemeType === 'State'}
                                onChange={handleSchemeTypeChange}
                                className="mr-2 text-blue-500 focus:ring-blue-500"
                            />
                            <span className="font-semibold">State Schemes</span>
                        </label>
                    </div>

                    {selectedSchemeType === 'State' && (
                        <select
                            value={selectedState}
                            onChange={handleStateChange}
                            className="border border-white/20 rounded-md px-3 py-2 bg-white/10 text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                            <option value="" className="bg-gray-800 text-white">-- Select State --</option>
                            {availableStates.map((state) => (
                                <option key={state} value={state} className="bg-gray-800 text-white">
                                    {state}
                                </option>
                            ))}
                        </select>
                    )}
                </div>

                {/* Results Summary */}
                {!loading && !error && (
                    <div className="mb-4">
                        <p className="text-gray-400 text-sm">
                            Showing {filteredSchemes.length} scheme{filteredSchemes.length !== 1 ? 's' : ''}
                            {selectedSchemeType !== 'all' && ` in ${selectedSchemeType} category`}
                            {selectedState && ` for ${selectedState}`}
                        </p>
                    </div>
                )}
            </div>

            {/* Loading State */}
            {loading && (
                <div className="flex items-center justify-center py-12">
                    <div className="bg-white/5 backdrop-blur-md rounded-lg p-8 border border-white/10">
                        <div className="flex items-center space-x-4">
                            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-400"></div>
                            <p className="text-gray-400 text-lg">Loading schemes...</p>
                        </div>
                    </div>
                </div>
            )}

            {/* Error State */}
            {error && !loading && (
                <div className="flex items-center justify-center py-12">
                    <div className="bg-red-500/10 backdrop-blur-md rounded-lg p-8 border border-red-500/20">
                        <p className="text-red-400 text-lg text-center">{error}</p>
                        <div className="mt-4 text-center">
                            <button
                                onClick={fetchLoanSchemes}
                                className="bg-red-600/40 hover:bg-red-600/60 text-red-200 px-4 py-2 rounded-md text-sm font-medium transition-all duration-200"
                            >
                                Retry Loading
                            </button>
                        </div>
                    </div>
                </div>
            )}

            {/* Schemes Content */}
            {!loading && !error && (
                <div className="grid grid-cols-1 gap-6">
                    {renderGroupedSchemes()}
                </div>
            )}
        </div>
    );
};

export default LoanManagement;