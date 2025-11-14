'use client';

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

export default function ProfileFields({ value, onChange }) {
  const handleChange = (field) => (event) => {
    onChange({ ...value, [field]: event.target.value });
  };

  return (
    <Stack spacing={3}>
      <TextField
        label="Age"
        type="number"
        value={value.age || ''}
        onChange={handleChange('age')}
        fullWidth
        inputProps={{ min: 0, max: 150 }}
        helperText="Enter your age"
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
    </Stack>
  );
}
