<script setup lang="ts">
    import EmptyState from './EmptyState.vue';
    import LoadingState from './LoadingState.vue';
    import PredictionResult from './PredictionResult.vue';
    import ErrorState from './ErrorState.vue';
    import type { ResultState } from '@/constants/api';

    defineProps<{ result: ResultState }>()
</script>
<template>
    <div class="w-full max-w-2xl flex flex-col items-center justify-center p-6 min-h-fit gap-6 bg-gray-100 rounded-xl border-2 border-gray-400">
        <h2 class="w-full font-bold">Prediction result</h2>
        <EmptyState v-if="result.status === 'empty'"></EmptyState>
        <LoadingState v-else-if="result.status === 'loading'"></LoadingState>

        <PredictionResult v-else-if="result.status === 'success'" :result ="result"></PredictionResult>

        <ErrorState v-else :result="result"></ErrorState>
    </div>
</template>
