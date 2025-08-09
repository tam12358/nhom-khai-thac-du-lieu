
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
  searchResults:SearchResult = new SearchResult();
  model:DataModel = new DataModel();
  protected readonly title = signal('rfUI');
  constructor(private http: HttpClient) { 
    this.search();
  }
  reranking(){
    this.searchResults.data = this.searchResults.data.sort((a, b) => b.score1 - a.score1);
  }
  normal(){
    this.searchResults.data = this.searchResults.data.sort((a, b) => b.score - a.score);
  }
  search(){
    console.log(this.searchResults);
    let rdocIds = this.searchResults.data.filter(x=>x.isChecked).map(x=>x.id);
    const dataToSend = { term: this.model.term, rdoc: rdocIds,offset: 0,limit: Number(this.model.topK), isReranking:this.model.isReranking };
    this.searchResults.total = 0;
    this.searchResults.time = 0;
    this.model.isSearch = true;
    this.http.post('http://127.0.0.1:5000/search', (dataToSend)).subscribe(
      response => {
        this.model.isSearch = false;
        console.log('POST request successful:', response);
        this.searchResults.data = [];
        var data = (response as any);
        var rows = data.data;
        this.searchResults.time = data.time;
        this.searchResults.total = data.total;
        this.searchResults.hit = data.hit;
        for(let index in rows){
          this.searchResults.data.push(new SearchItemResult(Number(rows[index].id),rows[index].content, Number(rows[index].score), Number(rows[index].score1)));
        }
        console.log(this.searchResults);
        
      },
      error => {
        console.error('Error during POST request:', error);
        this.searchResults.data = Array<SearchItemResult>();
      }
    );
  }

}
class SearchResult{
    data: Array<SearchItemResult> = [];
    time: any;
    total: number;
    hit: number = 0;
}
class SearchItemResult{
  constructor(_docId:number,_content:string, _score: number = 0, _score1: number = 0)
  {
    this.id = _docId;
    this.content = _content;
    this.short = this.content.substring(0,200);
    this.score = _score;
    this.score1 = _score1;
  }
  id : number | undefined;
  content: string = "";
  short: string = "";
  score: number = 0;
  score1: number = 0;
  isChecked:boolean = false;
}
class DataModel{
  term:string = "bóng đá";
  isReranking = false;
  topK: number = 20;
  isSearch: boolean = false;
}