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
        if (isElectron()) {
          const data = await FileAdapter.loadProfile();
          if (data) {
            if (typeof data.age !== 'undefined' && !data.dateOfBirth) {
              // Legacy profile detected. Clear all app data (test-only) per request.
              try {
                await FileAdapter.resetAppData();
              } catch {}
              setProfile({
                dateOfBirth: '',
                sex: '',
                raceEthnicity: '',
                hasCompletedOnboarding: false,
              });
            } else {
              const normalized = {
                dateOfBirth: data.dateOfBirth || '',
                sex: data.sex || '',
                raceEthnicity: data.raceEthnicity || '',
                hasCompletedOnboarding: false,
              };
              setProfile(normalized);
            }
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
    setProfile((prev) => ({ ...prev, ...newProfile }));
    // Attempt to persist
    if (isElectron()) {
      try {
        // Persist only known fields (exclude legacy 'age')
        const toSave = {
          dateOfBirth: (newProfile.dateOfBirth ?? profile.dateOfBirth) || '',
          sex: (newProfile.sex ?? profile.sex) || '',
          raceEthnicity: (newProfile.raceEthnicity ?? profile.raceEthnicity) || '',
          hasCompletedOnboarding:
            newProfile.hasCompletedOnboarding ?? profile.hasCompletedOnboarding ?? false,
        };
        await FileAdapter.saveProfile(toSave);
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
