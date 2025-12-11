'use client';

import { useEffect } from 'react';
import { TextField, MenuItem, Stack } from '@mui/material';

const SEX_OPTIONS = [
  { value: 'male', label: 'Male' },
  { value: 'female', label: 'Female' },
  { value: 'prefer-not-to-say', label: 'Prefer not to say' },
];

const RACE_ETHNICITY_OPTIONS = [
  { value: 'american-indian-alaska-native', label: 'American Indian or Alaska Native' },
  { value: 'asian', label: 'Asian' },
  { value: 'black-african-american', label: 'Black or African American' },
  { value: 'hispanic-latino', label: 'Hispanic or Latino' },
  { value: 'native-hawaiian-pacific-islander', label: 'Native Hawaiian or Other Pacific Islander' },
  { value: 'white', label: 'White' },
  { value: 'two-or-more', label: 'Two or More Races' },
  { value: 'other', label: 'Other' },
  { value: 'prefer-not-to-say', label: 'Prefer not to say' },
];

const COUNTRY_OPTIONS = [
  { value: 'us', label: 'United States' },
  { value: 'ca', label: 'Canada' },
  { value: 'mx', label: 'Mexico' },
  { value: 'uk', label: 'United Kingdom' },
  { value: 'de', label: 'Germany' },
  { value: 'fr', label: 'France' },
  { value: 'it', label: 'Italy' },
  { value: 'es', label: 'Spain' },
  { value: 'jp', label: 'Japan' },
  { value: 'cn', label: 'China' },
  { value: 'in', label: 'India' },
  { value: 'br', label: 'Brazil' },
  { value: 'au', label: 'Australia' },
  { value: 'other', label: 'Other' },
  { value: 'prefer-not-to-say', label: 'Prefer not to say' },
];

export default function ProfileFields({ value, onChange, onDateValidityChange }) {
  const handleChange = (field) => (event) => {
    onChange({ ...value, [field]: event.target.value });
  };

  const today = new Date();
  const maxDate = today.toISOString().split('T')[0];
  const minDateObj = new Date(today);
  minDateObj.setFullYear(minDateObj.getFullYear() - 200);
  const minDate = minDateObj.toISOString().split('T')[0];
  const dateOfBirth = value.dateOfBirth || '';
  const isDateInvalid =
    Boolean(dateOfBirth) && (dateOfBirth < minDate || dateOfBirth > maxDate);

  useEffect(() => {
    if (onDateValidityChange) {
      onDateValidityChange(!isDateInvalid);
    }
  }, [isDateInvalid, onDateValidityChange]);

  return (
    <Stack spacing={3}>
      <TextField
        label="Date of Birth"
        type="date"
        value={dateOfBirth}
        onChange={handleChange('dateOfBirth')}
        fullWidth
        InputLabelProps={{ shrink: true }}
        inputProps={{ min: minDate, max: maxDate }}
        error={isDateInvalid}
        helperText={isDateInvalid ? 'Invalid date' : undefined}
      />

      <TextField
        label="Sex"
        select
        value={value.sex || ''}
        onChange={handleChange('sex')}
        fullWidth
        helperText="Select your sex"
      >
        {SEX_OPTIONS.map((option) => (
          <MenuItem key={option.value} value={option.value}>
            {option.label}
          </MenuItem>
        ))}
      </TextField>

      <TextField
        label="Race/Ethnicity"
        select
        value={value.raceEthnicity || ''}
        onChange={handleChange('raceEthnicity')}
        fullWidth
        helperText="Select your race/ethnicity"
      >
        {RACE_ETHNICITY_OPTIONS.map((option) => (
          <MenuItem key={option.value} value={option.value}>
            {option.label}
          </MenuItem>
        ))}
      </TextField>

      <TextField
        label="Country of Origin"
        select
        value={value.country || ''}
        onChange={handleChange('country')}
        fullWidth
        helperText="Select your country of origin"
      >
        {COUNTRY_OPTIONS.map((option) => (
          <MenuItem key={option.value} value={option.value}>
            {option.label}
          </MenuItem>
        ))}
      </TextField>
    </Stack>
  );
}
