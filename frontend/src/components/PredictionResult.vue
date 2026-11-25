<script setup lang="ts">
import { computed } from 'vue';
import { CLASS_BY_ID, type ClassInfo, isClassId } from '@/constants/labels';
import { type ResultState } from '@/constants/api';
import UseWarning from './UseWarning.vue';

type SuccessResult = Extract<ResultState, { status: 'success' }>

const props = defineProps<{ result: SuccessResult }>()

const classInfo = computed<ClassInfo | null>(() => {
  const id = props.result.prediction.label
  return isClassId(id) ? CLASS_BY_ID[id] : null       // unknown label → null
})

// 0.5 to 1 → "50%" to "100%"
const confidencePct = computed(() => Math.round(props.result.prediction.confidence * 100))
</script>
<template>
    <div class="w-full flex flex-col gap-3">
        <div class="w-full bg-sky-50 px-4 py-6 flex flex-col gap-2 rounded-lg border-1 border-sky-500">
            <h2 class="text-xs font-medium text-sky-800">PREDICTION</h2>
            <p class="text-4xl font-bold">{{ classInfo?.name ?? result.prediction.label }}</p>
            <p class="text-sky-800">{{ classInfo?.description }}</p>

            <div class="flex flex-col gap-1 mt-2">
                <div class="flex items-baseline justify-between text-sm">
                    <span>Confidence</span>
                    <span class="font-bold tabular-nums">{{ confidencePct }}%</span>
                </div>
                <div
                    class="h-2.5 w-full rounded-full bg-sky-100 overflow-hidden"
                    role="meter"
                    aria-label="Confidence"
                    aria-valuemin="0"
                    aria-valuemax="100"
                    :aria-valuenow="confidencePct"
                >
                    <div class="h-full rounded-full bg-sky-700" :style="{ width: `${confidencePct}%` }"></div>
                </div>
                <p class="text-xs text-gray-500">50% means the model cannot tell; 100% means it is certain.</p>
            </div>

            <div class="w-full h-[1px] bg-gray-300"></div>
            <p class="text-xs text-gray-500">
                {{ result.prediction.analysed_seconds }} seconds analyzed · {{ result.prediction.processing_ms }} ms
            </p>
        </div>
        <UseWarning></UseWarning>
    </div>
</template>
