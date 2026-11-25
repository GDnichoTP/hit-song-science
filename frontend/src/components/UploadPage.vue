<script setup lang="ts">
import { ref, watch } from 'vue';
import UploadPanel from './UploadPanel.vue';
import ResultPanel from './ResultPanel.vue';
import { predict, ApiError, type ResultState } from '@/constants/api';
import { ERRORS } from '@/constants/errors';

const file = ref<File | null>(null)
const explicit = ref(false)
const result = ref<ResultState>({ status: 'empty' })

// A shown result belongs to one file and one explicit setting
watch([file, explicit], () => {
    result.value = { status: 'empty' }
})

async function upload() {
    const sent = file.value
    const sentExplicit = explicit.value
    if (!sent) return
    const isStale = () => file.value !== sent || explicit.value !== sentExplicit

    result.value = { status: 'loading' }
    try {
        const prediction = await predict(sent, sentExplicit)
        if (isStale()) return
        result.value = { status: 'success', prediction }
    } catch (e) {
        if (isStale()) return
        result.value = { status: 'error', error: e instanceof ApiError ? e.info : ERRORS.UNKNOWN_ERROR }
    }
}

</script>

<template>
    <div class="flex-1 flex justify-center items-center p-4 md:mx-8 md:p-8">
        <div class="grid grid-cols-1 md:grid-cols-2 gap-3 w-full max-w-6xl">
            <div class="flex justify-center">
                <UploadPanel v-model:file="file" v-model:explicit="explicit" @classify="upload"></UploadPanel>
            </div>
            <div class="flex justify-center">
                <ResultPanel :result="result"></ResultPanel>
            </div>
        </div>
    </div>
</template>
