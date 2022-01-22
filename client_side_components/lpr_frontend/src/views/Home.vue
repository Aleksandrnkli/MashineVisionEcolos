<template>
  <div class="logout">
    <Button 
    :text="'Logout'"
    @click="logout" 
    />
    <Filter 
    @reset-filters="resetFilters"
    @do-search="doSearch"
    />
  </div>
  <Detections 
  @toggle-wrong="toggleWrong"
  :detections="detections"
  />
  <PageBar 
  @previous-page="prevPage"
  @next-page="nextPage"
  :pages="pages" />
</template>

<script>
import Detections from '../components/Detections'
import PageBar from '../components/PageBar'
import Filter from '../components/Filter'
import Button from '../components/Button'

export default {
  name: 'Home',
  components: {
    Detections,
    PageBar,
    Filter,
    Button
  },

  data() {
    return {
      detections: [],
      pages: {},
      pageNum: null,
    }
  },

  props: {
    rest_server: String,
  },

  methods: {
    async fetchDetections(postfix = '') {
      const res = await fetch(this.rest_server + '/api/lpr/' + postfix, {
        credentials: 'omit',
        headers: {
          'Content-type': 'application/json',
          'Authorization': 'Token ' + localStorage.getItem('authToken'),
        },
      })

      const data = await res.json()

      return data
    },

    async fetchDetection(id) {
      const res = await fetch(this.rest_server + `/api/lpr/${id}/`, {
        credentials: 'omit',
        headers: {
          'Content-type': 'application/json',
          'Authorization': 'Token ' + localStorage.getItem('authToken'),
        },
      })

      const data = await res.json()

      return data
    },

    async getPages() {
      const res = await this.fetchDetections()
      let pageSize = res['results'].length
      if (res['count'] % pageSize !== 0) {
        this.pageNum = Math.floor(res['count'] / pageSize) + 1 
      } else {
        this.pageNum = Math.floor(res['count'] / pageSize)
      }
    },
    
    async displayDetections(postfix) {
      let res = null
      if (!this.pageNum) {
        await this.getPages()
        console.log(this.pageNum)
        res = await this.fetchDetections(`?page=${this.pageNum}`)
      } else {
        res = await this.fetchDetections(postfix)
      }

      this.detections = res["results"].reverse()
      // Here we swap next & prev 'cause we want to navigate
      // from new detections to older one
      this.pages = { previous: res["next"], next: res["previous"]}
    },

    async doSearch(searchParams) {
      const paramsBegin = '?'
      const paramsConn = '&'
      const timeConst = 'T00:00:00.000000%2B04%3A00'
      const licensePlate = 'license_plate=' + searchParams.licensePlate
      const fromDate = 'time_range_after=' + searchParams.fromDate + timeConst
      const toDate = 'time_range_before=' + searchParams.toDate + timeConst

      let postfix = paramsBegin + licensePlate + paramsConn + fromDate + paramsConn + toDate

      this.displayDetections(postfix)
    },

    async resetFilters() {
      this.pageNum = null
      this.displayDetections()
    },

    async nextPage() {
      this.displayDetections(this.pages.next.split("/lpr/")[1])
    },

    async prevPage() {
      this.displayDetections(this.pages.previous.split("/lpr/")[1])
    },

    async toggleWrong(id) {
      const detectionToToggle = await this.fetchDetection(id)
      const updDetection = {marked_as_error: !detectionToToggle.marked_as_error }

      const res = await fetch(this.rest_server + `/api/lpr/${id}/`, {
        method: 'PATCH',
        credentials: 'omit',
        headers: {
          'Content-type': 'application/json',
          'Authorization': 'Token ' + localStorage.getItem('authToken'),
        },
        body: JSON.stringify(updDetection),
      })

      const data = await res.json()

      this.detections = this.detections.map((detection) =>
        detection.id === id ? { ...detection, marked_as_error: data.marked_as_error } : detection
      )
    },

    async logout() {
      localStorage.removeItem('authToken')
      this.$router.push('/login')
    }
  },

  async created() {
    this.displayDetections()
  },

  beforeRouteEnter(from, to, next) {
    const authToken = localStorage.getItem('authToken');
    if (authToken) {
        next()
    } else {
        next('/login')
    }
  }
}
</script>

<style scoped>
  .logout Button {
    float: right;
    background: cornflowerblue;
    color: white;
    width: 15%;
  }
</style>