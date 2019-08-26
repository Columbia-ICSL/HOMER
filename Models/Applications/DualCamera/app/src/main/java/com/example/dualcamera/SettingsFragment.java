package com.example.dualcamera;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.widget.Toast;

import androidx.fragment.app.Fragment;
import androidx.preference.PreferenceFragmentCompat;
/**
 * A simple {@link Fragment} subclass.
 */
public class SettingsFragment extends PreferenceFragmentCompat implements SharedPreferences.OnSharedPreferenceChangeListener {

    @Override
    public void onCreatePreferences(Bundle savedInstanceState, String rootKey) {
        setPreferencesFromResource(R.xml.preferences, rootKey);
    }

    @Override
    public void onResume() {
        super.onResume();
        getPreferenceScreen().getSharedPreferences()
                .registerOnSharedPreferenceChangeListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        getPreferenceScreen().getSharedPreferences()
                .unregisterOnSharedPreferenceChangeListener(this);
    }

    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key)
    {
        try {
            if (Integer.parseInt(sharedPreferences.getString(SettingsActivity.KEY_PREF_MIN_DUR, "-1")) >
                    Integer.parseInt(sharedPreferences.getString(SettingsActivity.KEY_PREF_MAX_DUR, "-1"))) {
                Toast.makeText(getActivity(), "The max duration should be at least the min duration.", Toast.LENGTH_LONG).show();
            }
        } catch (NumberFormatException e) {
            Toast.makeText(getActivity(), "Please enter a non-negative integer.", Toast.LENGTH_LONG).show();
        }
    }
}
