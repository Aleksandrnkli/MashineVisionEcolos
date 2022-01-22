<template>
  <button @click="onClick()" :style="{ color: color, background: background }" class="btn">
    {{ text }}
  </button>
</template>

<script>
export default {
  name: 'Button',
  props: {
    text: String,
    color: String,
    background: String,
  },
  methods: {
    onClick() {
      this.$emit('btn-click')
    },
  },
}
</script>