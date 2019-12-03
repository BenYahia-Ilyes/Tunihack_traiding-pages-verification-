import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http'

@Component({
  selector: 'app-login',
  templateUrl: './login.component.html',
  styleUrls: ['./login.component.scss']
})
export class LoginComponent implements OnInit {
  Matricule=""
  Brand=""
  Sector=""

  serverData: JSON;
  saved: JSON;
  constructor(public http: HttpClient) {    }


  ngOnInit() {
  }


  postjson(){




    var json={Matricule:this.Matricule, Brand:this.Brand,Sector:this.Sector}
    this.http.post('http://127.0.0.1:5000/buisness',json).subscribe(data => {
      this.serverData = data as JSON;
  
   
  })

    
  }
  


}
