<template>
  <div :key="detection.id" v-for="detection in detections">
    <Detection  
      @toggle-wrong="$emit('toggle-wrong', detection.id)"
      :detection="detection" />
  </div>
</template>

<script>
import Detection from './Detection'

export default {
  name: 'Detections',
  props: {
    detections: Array,
  },
  components: {
    Detection,
  },
  emits: ['toggle-wrong'],
}
</script>