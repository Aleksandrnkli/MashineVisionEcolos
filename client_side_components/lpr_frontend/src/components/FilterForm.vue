<template>
  <form @submit="onSubmit" class="filter-form">
    <div class="form-lp">
      <label>License plate</label>
      <input type="text" v-model="licensePlate" name="licensePlate"/>
    </div>
    <div class="form-date">
      <label >Date range</label>
      <input type="date" v-model="fromDate" name="fromDate" id="fromDate">
      <input type="date" v-model="toDate" name="toDate" id="toDate">
    </div>
    <div class="search-submit">
      <input type="submit" value="Search" class="search-btn"/>
    </div>
  </form>
</template>

<script>
export default {
  name: 'FilterForm',
  data() {
    return {
      licensePlate: '',
      fromDate: '',
      toDate: ''
    }
  },
  methods: {
    onSubmit(e) {
      e.preventDefault()

      document.getElementById('fromDate').required = false
      document.getElementById('toDate').required = false

      if (!this.fromDate || !this.toDate) {
        document.getElementById('fromDate').required = true
        document.getElementById('toDate').required = true
        return
      }

      let dateParts = this.toDate.split('-')
      dateParts[2] = (parseInt(dateParts[2]) + 1).toString()
      this.toDate = `${dateParts[0]}-${dateParts[1]}-${dateParts[2]}`

      const newSearch = {
        licensePlate: this.licensePlate,
        fromDate: this.fromDate,
        toDate: this.toDate
      }

      this.$emit('do-search', newSearch)

      this.licensePlate = ''
      this.fromDate = ''
      this.toDate = ''
    },
  },
}
</script>

<style scoped>
.filter-form {
  margin-bottom: 40px;
}

.form-lp {
  margin: 20px 0;
}

label {
  display: block;
}

.form-lp input {
  width: 50%;
  height: 40px;
  margin: 5px;
  font-size: 17px;
  text-align: center;
}

.form-date input {
  width: 24%;
  height: 40px;
  margin: 5px;
  font-size: 17px;
  text-align: center;
}

.form-lp-check {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.form-lp {
  text-align: center;
  padding-top: 10px;
}

.form-date {
  text-align: center;
  padding-bottom: 10px;
}

.search-submit {
  text-align: center;
}

.search-submit .search-btn {
  display: inline-block;
  background: cornflowerblue;
  color: #fff;
  border: none;
  padding: 10px 20px;
  margin: 5px;
  border-radius: 5px;
  cursor: pointer;
  text-decoration: none;
  font-size: 15px;
  font-family: inherit;
}
</style>
