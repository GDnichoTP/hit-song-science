<script setup lang="ts">
    import FileDropZone from './FileDropZone.vue';
    import BasedButton from './BasedButton.vue';
    import PrivacyNote from './PrivacyNote.vue';
    import SelectedFile from './SelectedFile.vue';
    import AudioPreview from './AudioPreview.vue';

    import { useTemplateRef } from 'vue';

    const file = defineModel<File | null> ('file', {default:null})
    // Explicit lyrics cannot be measured from the audio, so the user says it
    const explicit = defineModel<boolean> ('explicit', {default:false})
    const dropZone = useTemplateRef('dropZone')

    const emit = defineEmits<{
        classify: []
    }>()

</script>

<template>
    <div class="flex flex-col max-w-2xl w-full p-6 h-fit space-y-6 bg-gray-100 rounded-xl border-2 border-gray-400">
        <h2 class="w-full font-bold">Upload a song clip</h2>
        <FileDropZone v-show="file === null" ref="dropZone" v-model:file="file"></FileDropZone>
        <SelectedFile
            v-if="file !== null"
            :file="file"
            @remove="file = null"
            @change="dropZone?.openPicker()"
        ></SelectedFile>
        <AudioPreview v-if="file !== null" :file="file"></AudioPreview>
        <label class="flex items-start gap-3 cursor-pointer select-none">
            <input v-model="explicit" type="checkbox" class="mt-1 size-4 accent-sky-700 cursor-pointer" />
            <span class="flex flex-col">
                <span>The song has explicit lyrics</span>
                <span class="text-xs text-gray-500">Marked "E" (explicit) on streaming services</span>
            </span>
        </label>
        <BasedButton :disabled="file === null" buttonLabel="Predict" @click="file !== null && emit('classify')"></BasedButton>
        <PrivacyNote noteLabel='The clip is sent to ReccoBeats, an outside audio analysis service, to measure its audio features. Files are only processed during the request and are not stored.'></PrivacyNote>
    </div>
</template>
