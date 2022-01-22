<template>
  <form @submit="onSubmit" class="login-form">
    <div class="form-control">
      <label>Login form</label>
      <input type="text" v-model="username" name="username" placeholder="Username"/>
      <input type="password" v-model="password" name="password" placeholder="Password"/>
      <input type="submit" value="Login" class="login-submit-btn" />
    </div>
  </form>
</template>

<script>
export default {
  name: 'Login',

  data() {
    return {
      username: '',
      password: ''
    }
  },

  props: {
    rest_server: String,
  },

  methods: {
    async onSubmit(e) {
      e.preventDefault()

      const res = await fetch(this.rest_server + `/api_token_auth/`, {
        method: 'POST',
        credentials: 'omit',
        headers: {
          'Content-type': 'application/x-www-form-urlencoded',
        },
        body: `username=${this.username}&password=${this.password}`
      })
        .then(res => {
          return res.json()
        })

      this.username = ''
      this.password = ''
      
      if (res['token']) {
        localStorage.setItem('authToken', res['token'])
        this.$router.push('/')
      } else {
        alert('Invalid password or/and username')
        const inputs = document.querySelectorAll("input");

        inputs.forEach((input) => {
          input.required = true;
        });
      }
    },
  }
} 
    
</script>

<style scoped>
.login-form {
  margin-bottom: 40px;
}

.form-control {
  margin: 20px 0;
  text-align: center;
}

.form-control label {
  display: block;
  font-size: 21px;
}

.form-control input {
  width: 50%;
  height: 30px;
  margin: 5px;
  padding: 3px 7px;
  font-size: 17px;
  text-align: center;
}

.login-submit-btn {
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