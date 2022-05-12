<template>
    <div class="pages">
        <Button
        v-if="pages.previous !== null"
        :text="'Prev page'"
        @click="$emit('previous-page')" 
        />
        <Button
        v-if="pages.next !== null"
        :text="'Next page'"
        @click="$emit('next-page')" 
        />
    </div>    
</template>

<script>
import Button from './Button'

export default {
    name: 'PageBar',
    components: {
        Button
    },
    props: {
        pages: Object,
    },
    emits: ['previous-page', 'next-page']
}
</script>

<style scoped>
    .pages {
        text-align: center;
    }
    .pages Button {
        display: inline-block;
        background: cornflowerblue;
        color: #fff;
        border: none;
        padding: 10px 20px;
        margin:8px;
        border-radius: 5px;
        cursor: pointer;
        text-decoration: none;
        font-size: 15px;
        font-family: inherit;
    }
</style>