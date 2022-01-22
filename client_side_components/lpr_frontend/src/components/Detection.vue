<template>
  <div
    @dblclick="$emit('toggle-wrong', detection.id)"
    :class="[detection.marked_as_error ? 'wrong' : '', 'detection']"
  >
    <h3>
        {{ detection.license_plate }}
    </h3>
    <p>Event time: 
        {{ detection.detection_time.split('.')[0].split('T')[0]}}
        {{ detection.detection_time.split('.')[0].split('T')[1]}}
    </p>
    <img v-bind:src="detection.image" alt="NOT FOUND">
  </div>
</template>

<script>
export default {
  name: 'Detection',
  props: {
    detection: Object,
  },
}
</script>

<style scope>
.detection {
  border: 2px solid steelblue;
  margin: 5px;
  padding: 10px 20px;
  cursor: pointer;
}

.detection.wrong {
  border-left: 8px solid red;
}

.detection h3 {
  display: flex;
  justify-content: space-between;
}

img {
    max-width: 100%;
    height: auto;
}
</style>