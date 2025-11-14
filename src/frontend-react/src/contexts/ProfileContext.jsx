'use client';

import { createContext, useContext, useState, useEffect } from 'react';
import FileAdapter from '@/services/adapters/fileAdapter';
import { isElectron } from '@/utils/config';

const ProfileContext = createContext();

export function ProfileProvider({ children }) {
  const [profile, setProfile] = useState({
    age: '',
    sex: '',
    raceEthnicity: '',
    hasCompletedOnboarding: false,
  });
  const [loading, setLoading] = useState(true);

  // Load profile on mount
  useEffect(() => {
    async function load() {
      try {
        if (isElectron()) {
          const data = await FileAdapter.loadProfile();
          if (data) setProfile({ hasCompletedOnboarding: false, ...data });
        }
      } catch (e) {
        console.warn('Failed to load profile:', e);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const updateProfile = async (newProfile) => {
    setProfile((prev) => ({ ...prev, ...newProfile }));
    // Attempt to persist
    if (isElectron()) {
      try {
        await FileAdapter.saveProfile({ ...profile, ...newProfile });
      } catch (e) {
        console.error('Failed to save profile:', e);
      }
    }
  };

  return (
    <ProfileContext.Provider value={{ profile, updateProfile, loading }}>
      {children}
    </ProfileContext.Provider>
  );
}

export function useProfile() {
  const context = useContext(ProfileContext);
  if (!context) {
    throw new Error('useProfile must be used within ProfileProvider');
  }
  return context;
}
