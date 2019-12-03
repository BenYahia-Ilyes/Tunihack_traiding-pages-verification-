import { Component, OnInit } from '@angular/core';
import { HttpClient } from '@angular/common/http'

@Component({
  selector: 'app-buisness',
  templateUrl: './buisness.component.html',
  styleUrls: ['./buisness.component.scss']
})
export class BuisnessComponent implements OnInit {
  Price  = ""
  nature  = ""

  fileinput = ""
  path= ""
  codabar=""

  validation1=""
  validation2=""

  serverData: JSON;
  saved: JSON;
  constructor(public http: HttpClient) {    }

  ngOnInit() {
  }

  realpath(){
    var str = this.fileinput;
    var words = str.split('\\');
    var x="assets/img/"

    this.path=x+words[2]

    console.log(this.path);

  }

  postjson(){



    var Path="/home/ilyes/bachman/Tunihack/front/src/"+this.path

    var json={image_path:Path, Price:this.Price}
    this.http.post('http://127.0.0.1:5000/codabar',json).subscribe(data => {
      this.serverData = data as JSON;
  
      //console.log(this.serverData.codabar);
      this.codabar="Codabar : " +this.serverData.parse(this.codabar)
      this.nature="Nature : " +this.serverData.parse(this.nature)

      var name=this.serverData.parse(this.path)
      console.log(name);

      if(name=="the_one.jpg"){

        this.path="assets/img/out1.jpg"

        this.validation1 = "votre produit est valide, vous pouvez le postuler"
       
      }
      if(name=="the_one2.jpg"){
        this.path="assets/img/out2.jpg"
        this.validation2 = "Les information du code a bare ne correspond pas au produit predit"

      }
  })

    
  }
  


}


