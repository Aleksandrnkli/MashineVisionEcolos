<template>
    <Button
      v-show="!filterMode"
      @btn-click="filterMode = true"
      :text="'Filter'"
    />
    <div v-show="filterMode">
      <Button
      @btn-click="$emit('reset-filters'); filterMode = false"
      :text="'Reset'"
      />
      <FilterForm 
      @do-search="doSearch"
      />
    </div>
</template>

<script>
import Button from './Button'
import FilterForm from './FilterForm'

export default {
    name: "Filter",
    components: {
        Button,
        FilterForm
    },
    data() {
        return {
            filterMode: false
        }
    },
    methods: {
        doSearch(newSearch) {
            this.$emit('do-search', newSearch)
        }
    },
    emits: ['do-search', 'reset-filters'],    
}
</script>

<style scoped>

</style>