const API_URL = process.env.REACT_APP_API_URL || process.env.REACT_APP_API_BASE || '';

export const fetchRooms = async () => {
  const res = await fetch(`${API_URL}/api/rooms`);
  return res.json();
};
