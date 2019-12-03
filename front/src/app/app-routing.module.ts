import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
//import { HttpModule } from '@angular/http';
import { HttpClientModule, HttpClient } from '@angular/common/http';
const routes: Routes = [];

@NgModule({
  imports: [RouterModule.forRoot(routes),   
         HttpClientModule,
    ],
  exports: [RouterModule]
})
export class AppRoutingModule { }
