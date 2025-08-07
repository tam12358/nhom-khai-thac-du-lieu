
import { Component, NgModule, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { HttpClient } from '@angular/common/http';
import { FormsModule } from '@angular/forms';
import { HttpClientModule } from '@angular/common/http';

export class AppModule { }

@Component({
  selector: 'app-root',
  imports: [RouterOutlet,FormsModule,HttpClientModule],
  templateUrl: './app.html',
  styleUrl: './app.css'
})

export class App {
  searchResults = Array<SearchResult>();
  model:DataModel = new DataModel();
  protected readonly title = signal('rfUI');
  constructor(private http: HttpClient) { 
    this.search();
  }
  search(){
    console.log(this.searchResults);
    let rdocIds = this.searchResults.filter(x=>x.isChecked).map(x=>x.id);
    const dataToSend = { term: this.model.term, rdoc: rdocIds,offset: 0,limit: 20 };
    this.http.post('http://127.0.0.1:5000/search', (dataToSend)).subscribe(
      response => {
        console.log('POST request successful:', response);
        this.searchResults=[];
        for(let index in response){
          this.searchResults.push(new SearchResult(Number(response[index].id),response[index].content));
        }
        console.log(this.searchResults);
        
      },
      error => {
        console.error('Error during POST request:', error);
        this.searchResults = Array<SearchResult>();
      }
    );
  }

}
class SearchResult{
  constructor(_docId:number,_content:string){
    this.id = _docId;
    this.content = _content;
    this.short = this.content.substring(0,150);
  }
  id : number | undefined;
  content: string = "";
  short: string = "";
  isChecked:boolean = false;
}
class DataModel{
  term:string = "";
}