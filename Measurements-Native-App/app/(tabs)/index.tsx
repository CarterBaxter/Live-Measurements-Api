import { Image } from 'expo-image';
import * as ImagePicker from 'expo-image-picker';
import { useState } from 'react';
import {
  ActivityIndicator,
  Alert,
  Pressable,
  ScrollView,
  Switch,
  StyleSheet,
  TextInput,
  View,
} from 'react-native';

import { ThemedText } from '@/components/themed-text';
import { ThemedView } from '@/components/themed-view';
import { uploadMeasurements, type MeasurementsResponse } from '@/lib/measurementsApi';

const ENDPOINT_URL = 'http://127.0.0.1:5001/live-measurements-test/us-central1/get_measurements';

export default function HomeScreen() {
  const [frontUri, setFrontUri] = useState<string | null>(null);
  const [sideUri, setSideUri] = useState<string | null>(null);
  const [heightCm, setHeightCm] = useState('');
  const [useDepth, setUseDepth] = useState(false);
  const [fastDepth, setFastDepth] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [response, setResponse] = useState<MeasurementsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pickImage = async (source: 'camera' | 'library', target: 'front' | 'side') => {
    const permissionResult =
      source === 'camera'
        ? await ImagePicker.requestCameraPermissionsAsync()
        : await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!permissionResult.granted) {
      Alert.alert('Permission required', 'Please grant permission to access your camera or photos.');
      return;
    }

    const result =
      source === 'camera'
        ? await ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            quality: 0.8,
          })
        : await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Images,
            quality: 0.8,
          });

    if (result.canceled) {
      return;
    }

    const uri = result.assets[0]?.uri;
    if (!uri) {
      return;
    }

    if (target === 'front') {
      setFrontUri(uri);
    } else {
      setSideUri(uri);
    }
  };

  const handleSubmit = async () => {
    if (!frontUri) {
      Alert.alert('Missing front photo', 'Please select or take a front photo to continue.');
      return;
    }

    const trimmedHeight = heightCm.trim();
    const parsedHeight = trimmedHeight ? Number(trimmedHeight) : undefined;
    if (trimmedHeight && Number.isNaN(parsedHeight)) {
      Alert.alert('Invalid height', 'Please enter a valid height in cm.');
      return;
    }

    setIsSubmitting(true);
    setError(null);
    setResponse(null);

    try {
      const payload = await uploadMeasurements({
        endpointUrl: ENDPOINT_URL,
        frontUri,
        leftSideUri: sideUri ?? undefined,
        heightCm: parsedHeight,
        useDepth,
        depthMaxDim: useDepth && fastDepth ? 640 : undefined,
      });
      setResponse(payload);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Request failed.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <ThemedText type="title">Get Measurements</ThemedText>
      <ThemedText style={styles.helperText}>
        Provide a front photo (required), optional side photo, and optional height in cm.
      </ThemedText>

      <ThemedView style={styles.section}>
        <ThemedText type="subtitle">Front photo (required)</ThemedText>
        <View style={styles.buttonRow}>
          <Pressable style={styles.button} onPress={() => pickImage('library', 'front')}>
            <ThemedText style={styles.buttonText}>Choose Photo</ThemedText>
          </Pressable>
          <Pressable style={styles.button} onPress={() => pickImage('camera', 'front')}>
            <ThemedText style={styles.buttonText}>Take Photo</ThemedText>
          </Pressable>
        </View>
        {frontUri ? <Image source={{ uri: frontUri }} style={styles.preview} /> : null}
      </ThemedView>

      <ThemedView style={styles.section}>
        <ThemedText type="subtitle">Side photo (optional)</ThemedText>
        <View style={styles.buttonRow}>
          <Pressable style={styles.button} onPress={() => pickImage('library', 'side')}>
            <ThemedText style={styles.buttonText}>Choose Photo</ThemedText>
          </Pressable>
          <Pressable style={styles.button} onPress={() => pickImage('camera', 'side')}>
            <ThemedText style={styles.buttonText}>Take Photo</ThemedText>
          </Pressable>
        </View>
        {sideUri ? <Image source={{ uri: sideUri }} style={styles.preview} /> : null}
      </ThemedView>

      <ThemedView style={styles.section}>
        <ThemedText type="subtitle">Height (cm, optional)</ThemedText>
        <TextInput
          style={styles.input}
          keyboardType="numeric"
          placeholder="e.g. 170"
          value={heightCm}
          onChangeText={setHeightCm}
        />
      </ThemedView>

      <ThemedView style={styles.section}>
        <ThemedText type="subtitle">Depth estimation (optional)</ThemedText>
        <View style={styles.toggleRow}>
          <ThemedText>Enable depth for higher accuracy</ThemedText>
          <Switch value={useDepth} onValueChange={setUseDepth} />
        </View>
        <ThemedText style={styles.helperText}>
          Depth can be slower and may time out on large images.
        </ThemedText>
      </ThemedView>

      {useDepth ? (
        <ThemedView style={styles.section}>
          <ThemedText type="subtitle">Depth speed (optional)</ThemedText>
          <View style={styles.toggleRow}>
            <ThemedText>Fast mode (downscale input)</ThemedText>
            <Switch value={fastDepth} onValueChange={setFastDepth} />
          </View>
          <ThemedText style={styles.helperText}>
            Fast mode reduces resolution to 640px on the long edge.
          </ThemedText>
        </ThemedView>
      ) : null}

      <Pressable style={[styles.submitButton, isSubmitting && styles.submitButtonDisabled]} onPress={handleSubmit} disabled={isSubmitting}>
        {isSubmitting ? (
          <ActivityIndicator color="#ffffff" />
        ) : (
          <ThemedText style={styles.submitText}>Submit</ThemedText>
        )}
      </Pressable>

      {error ? (
        <ThemedText style={styles.errorText}>{error}</ThemedText>
      ) : null}

      {response ? (
        <View style={styles.responseBox}>
          <ThemedText type="subtitle">Response</ThemedText>
          <ThemedText style={styles.responseText}>
            {JSON.stringify(response, null, 2)}
          </ThemedText>
        </View>
      ) : null}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flexGrow: 1,
    padding: 20,
    gap: 16,
  },
  helperText: {
    opacity: 0.7,
  },
  section: {
    gap: 10,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
    flexWrap: 'wrap',
  },
  toggleRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    gap: 12,
  },
  button: {
    backgroundColor: '#1f6feb',
    paddingVertical: 10,
    paddingHorizontal: 14,
    borderRadius: 8,
  },
  buttonText: {
    color: '#ffffff',
    fontWeight: '600',
  },
  preview: {
    height: 180,
    borderRadius: 12,
    marginTop: 10,
  },
  input: {
    borderWidth: 1,
    borderColor: '#c7c7c7',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 10,
    backgroundColor: '#ffffff',
  },
  submitButton: {
    backgroundColor: '#0f62fe',
    paddingVertical: 14,
    borderRadius: 10,
    alignItems: 'center',
  },
  submitButtonDisabled: {
    opacity: 0.7,
  },
  submitText: {
    color: '#ffffff',
    fontWeight: '700',
  },
  responseBox: {
    borderWidth: 1,
    borderColor: '#d4d4d4',
    borderRadius: 10,
    padding: 12,
    gap: 8,
  },
  responseText: {
    fontFamily: 'monospace',
    fontSize: 12,
  },
  errorText: {
    color: '#b42318',
  },
});
