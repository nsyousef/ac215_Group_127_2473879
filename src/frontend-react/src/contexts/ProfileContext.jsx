'use client';

import { createContext, useContext, useState, useEffect } from 'react';
import FileAdapter from '@/services/adapters/fileAdapter';
import { isElectron } from '@/utils/config';

const ProfileContext = createContext();

export function ProfileProvider({ children }) {
  const [profile, setProfile] = useState({
    dateOfBirth: '',
    sex: '',
    raceEthnicity: '',
    hasCompletedOnboarding: false,
  });
  const [loading, setLoading] = useState(true);

  // Load profile on mount
  useEffect(() => {
    async function load() {
      try {
        if (isElectron() && window.electronAPI?.loadDemographics) {
          const data = await window.electronAPI.loadDemographics();
          if (data && Object.keys(data).length > 0) {
            // Map demographics.json schema to profile
            const normalized = {
              dateOfBirth: data.DOB || '',
              sex: data.Sex || '',
              raceEthnicity: data.Race || '',
              hasCompletedOnboarding: data.hasCompletedOnboarding || false,
            };
            setProfile(normalized);
          }
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
    const updated = { ...profile, ...newProfile };
    setProfile(updated);
    
    // Attempt to persist
    if (isElectron() && window.electronAPI?.saveDemographics) {
      try {
        // Map profile fields to demographics.json schema
        const demographics = {
          DOB: updated.dateOfBirth || '',
          Sex: updated.sex || '',
          Race: updated.raceEthnicity || '',
          hasCompletedOnboarding: updated.hasCompletedOnboarding || false,
        };
        await window.electronAPI.saveDemographics(demographics);
      } catch (e) {
        console.error('Failed to save demographics:', e);
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
